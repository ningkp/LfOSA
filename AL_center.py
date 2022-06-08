import os
import sys
import argparse
import datetime
import time
import os.path as osp
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import numpy as np
import random

import torch
import torch.nn as nn
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from resnet import resnet18, resnet34, resnet50, resnet101, resnet152
import Sampling

import datasets
import models
import torchvision.models as torch_models
import pickle
from utils import AverageMeter, Logger
from center_loss import CenterLoss
from sklearn.mixture import GaussianMixture


parser = argparse.ArgumentParser("Center Loss Example")
# dataset
parser.add_argument('-d', '--dataset', type=str, default='cifar10', choices=['mnist', 'cifar100', 'cifar10'])
parser.add_argument('-j', '--workers', default=0, type=int,
                    help="number of data loading workers (default: 4)")
# optimization
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr-model', type=float, default=0.01, help="learning rate for model")
parser.add_argument('--lr-cent', type=float, default=0.5, help="learning rate for center loss")
parser.add_argument('--weight-cent', type=float, default=1, help="weight for center loss")
parser.add_argument('--max-epoch', type=int, default=100)
parser.add_argument('--max-query', type=int, default=10)
parser.add_argument('--query-batch', type=int, default=1500)
parser.add_argument('--query-strategy', type=str, default='AV_based2', choices=['random', 'uncertainty', 'AV_based', 'AV_uncertainty', 'AV_based2', 'Max_AV'])
parser.add_argument('--stepsize', type=int, default=20)
parser.add_argument('--gamma', type=float, default=0.5, help="learning rate decay")
# model
parser.add_argument('--model', type=str, default='resnet18')
# misc
parser.add_argument('--eval-freq', type=int, default=100)
parser.add_argument('--print-freq', type=int, default=50)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--save-dir', type=str, default='log')
parser.add_argument('--plot', action='store_true', help="whether to plot features for every epoch")
# openset
parser.add_argument('--is-filter', type=bool, default=True)
parser.add_argument('--is-mini', type=bool, default=True)
parser.add_argument('--known-class', type=int, default=20)
parser.add_argument('--init-percent', type=int, default=16)

args = parser.parse_args()

def main():
    torch.manual_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    use_gpu = torch.cuda.is_available()
    if args.use_cpu: use_gpu = False

    sys.stdout = Logger(osp.join(args.save_dir, 'log_' + args.dataset + '.txt'))

    if use_gpu:
        print("Currently using GPU: {}".format(args.gpu))
        cudnn.benchmark = True
        torch.cuda.manual_seed_all(args.seed)
    else:
        print("Currently using CPU")

    print("Creating dataset: {}".format(args.dataset))
    dataset = datasets.create(
        name=args.dataset, known_class_=args.known_class, init_percent_=args.init_percent,
        batch_size=args.batch_size, use_gpu=use_gpu,
        num_workers=args.workers, is_filter=args.is_filter, is_mini=args.is_mini,
    )

    trainloader, testloader, unlabeledloader = dataset.trainloader, dataset.testloader, dataset.unlabeledloader
    labeled_ind_train, unlabeled_ind_train = dataset.labeled_ind_train, dataset.unlabeled_ind_train

    print("Creating model: {}".format(args.model))
    # model = models.create(name=args.model, num_classes=dataset.num_classes)
    #
    # if use_gpu:
    #     model = nn.DataParallel(model).cuda()
    #
    # criterion_xent = nn.CrossEntropyLoss()
    # criterion_cent = CenterLoss(num_classes=dataset.num_classes, feat_dim=2, use_gpu=use_gpu)
    # optimizer_model = torch.optim.SGD(model.parameters(), lr=args.lr_model, weight_decay=5e-04, momentum=0.9)
    # optimizer_centloss = torch.optim.SGD(criterion_cent.parameters(), lr=args.lr_cent)
    #
    # if args.stepsize > 0:
    #     scheduler = lr_scheduler.StepLR(optimizer_model, step_size=args.stepsize, gamma=args.gamma)

    start_time = time.time()

    Acc = {}
    Err = {}
    Precision = {}
    Recall = {}
    for query in tqdm(range(args.max_query)):
        # Model initialization
        if args.model == "cnn":
            model = models.create(name=args.model, num_classes=dataset.num_classes)
        elif args.model == "resnet18":
            model = resnet18(num_classes=dataset.num_classes)
        elif args.model == "resnet34":
            model = resnet34(num_classes=dataset.num_classes)
        elif args.model == "resnet50":
            model = resnet50(num_classes=dataset.num_classes)

        if use_gpu:
            model = nn.DataParallel(model).cuda()

        criterion_xent = nn.CrossEntropyLoss()
        criterion_cent = CenterLoss(num_classes=dataset.num_classes, feat_dim=2, use_gpu=use_gpu)
        optimizer_model = torch.optim.SGD(model.parameters(), lr=args.lr_model, weight_decay=5e-04, momentum=0.9)
        optimizer_centloss = torch.optim.SGD(criterion_cent.parameters(), lr=args.lr_cent)

        if args.stepsize > 0:
            scheduler = lr_scheduler.StepLR(optimizer_model, step_size=args.stepsize, gamma=args.gamma)

        # Model training 
        for epoch in tqdm(range(args.max_epoch)):
        # for epoch in tqdm(range(0)):
        #     print("==> Epoch {}/{}".format(epoch+1, args.max_epoch))
            train(model, criterion_xent, criterion_cent,
                  optimizer_model, optimizer_centloss,
                  trainloader, use_gpu, dataset.num_classes, epoch)

            if args.stepsize > 0: scheduler.step()

            if args.eval_freq > 0 and (epoch+1) % args.eval_freq == 0 or (epoch+1) == args.max_epoch:
                print("==> Test")
                acc, err = test(model, testloader, use_gpu, dataset.num_classes, epoch)
                print("Accuracy (%): {}\t Error rate (%): {}".format(acc, err))

        # Record results
        acc, err = test(model, testloader, use_gpu, dataset.num_classes, args.max_epoch)
        Acc[query], Err[query] = float(acc), float(err)
        # Query samples and calculate precision and recall
        queryIndex = []
        if args.query_strategy == "random":
            queryIndex, invalidIndex, Precision[query], Recall[query] = Sampling.random_sampling(args, unlabeledloader, len(labeled_ind_train), model, use_gpu)
        elif args.query_strategy == "uncertainty":
            queryIndex, invalidIndex, Precision[query], Recall[query] = Sampling.uncertainty_sampling(args, unlabeledloader, len(labeled_ind_train), model, use_gpu)
        elif args.query_strategy == "AV_based":
            queryIndex, invalidIndex, Precision[query], Recall[query] = Sampling.AV_sampling(args, unlabeledloader, len(labeled_ind_train), model, use_gpu)
        elif args.query_strategy == "AV_uncertainty":
            queryIndex, invalidIndex, Precision[query], Recall[query] = Sampling.AV_uncertainty_sampling(args, unlabeledloader, len(labeled_ind_train), model, use_gpu)
        elif args.query_strategy == "AV_based2":
            queryIndex, invalidIndex, Precision[query], Recall[query] = Sampling.AV_sampling2(args, trainloader, unlabeledloader, len(labeled_ind_train), model, use_gpu)
        elif args.query_strategy == "Max_AV":
            queryIndex, invalidIndex, Precision[query], Recall[query] = Sampling.Max_AV_sampling(args, unlabeledloader, len(labeled_ind_train), model, use_gpu)
        # Update labeled, unlabeled and invalid set
        unlabeled_ind_train = list(set(unlabeled_ind_train)-set(queryIndex))
        labeled_ind_train = list(labeled_ind_train) + list(queryIndex)
        print("Query Strategy: "+args.query_strategy+" | Query Batch: "+str(args.query_batch)+" | Valid Query Nums: "+str(len(queryIndex))+" | Query Precision: "+str(Precision[query])+" | Query Recall: "+str(Recall[query])+" | Training Nums: "+str(len(labeled_ind_train))+" | Unalebled Nums: "+str(len(unlabeled_ind_train)))
        dataset = datasets.create(
            name=args.dataset, known_class_=args.known_class, init_percent_=args.init_percent,
            batch_size=args.batch_size, use_gpu=use_gpu,
            num_workers=args.workers, is_filter=args.is_filter, is_mini=args.is_mini,
            unlabeled_ind_train=unlabeled_ind_train, labeled_ind_train=labeled_ind_train,
        )
        trainloader, testloader, unlabeledloader = dataset.trainloader, dataset.testloader, dataset.unlabeledloader
        labeled_ind_train, unlabeled_ind_train = dataset.labeled_ind_train, dataset.unlabeled_ind_train

    ## Save results
    with open("log_AL/"+args.model+"_"+args.dataset+"_known"+str(args.known_class)+"_init"+str(args.init_percent)+"_batch"+str(args.query_batch)+"_seed"+str(args.seed)+"_"+args.query_strategy+".pkl", 'wb') as f:
        data = {'Acc': Acc, 'Err': Err, 'Precision': Precision, 'Recall': Recall}
        pickle.dump(data, f)
    f.close()
    ## Save model
    if args.is_mini:
        torch.save(model, "save_model/AL_center_"+args.dataset+"30_mini_query"+str(args.max_query)+"_batch"+str(args.query_batch)+".pt")
    else:
        torch.save(model, "save_model/AL_center_"+args.dataset+"30_query"+str(args.max_query)+"_batch"+str(args.query_batch)+".pt")
    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))


def calculate_precision_recall():
    precision, recall = 0, 0
    return precision, recall

def train(model, criterion_xent, criterion_cent,
          optimizer_model, optimizer_centloss,
          trainloader, use_gpu, num_classes, epoch):
    model.train()
    xent_losses = AverageMeter()
    cent_losses = AverageMeter()
    losses = AverageMeter()
    
    if args.plot:
        all_features, all_labels = [], []

    for batch_idx, (index, (data, labels)) in enumerate(trainloader):
        if use_gpu:
            data, labels = data.cuda(), labels.cuda()
        features, outputs = model(data)
        loss_xent = criterion_xent(outputs, labels)
        loss_cent = criterion_cent(features, labels)
        loss_cent *= args.weight_cent
        loss = loss_xent + loss_cent
        optimizer_model.zero_grad()
        optimizer_centloss.zero_grad()
        loss.backward()
        optimizer_model.step()
        # by doing so, weight_cent would not impact on the learning of centers
        if args.weight_cent > 0.0:
            for param in criterion_cent.parameters():
                param.grad.data *= (1. / args.weight_cent)
            optimizer_centloss.step()
        
        losses.update(loss.item(), labels.size(0))
        xent_losses.update(loss_xent.item(), labels.size(0))
        cent_losses.update(loss_cent.item(), labels.size(0))

        if args.plot:
            if use_gpu:
                all_features.append(features.data.cpu().numpy())
                all_labels.append(labels.data.cpu().numpy())
            else:
                all_features.append(features.data.numpy())
                all_labels.append(labels.data.numpy())

        if (batch_idx+1) % args.print_freq == 0:
            print("Batch {}/{}\t Loss {:.6f} ({:.6f}) XentLoss {:.6f} ({:.6f}) CenterLoss {:.6f} ({:.6f})" \
                  .format(batch_idx+1, len(trainloader), losses.val, losses.avg, xent_losses.val, xent_losses.avg, cent_losses.val, cent_losses.avg))

    if args.plot:
        all_features = np.concatenate(all_features, 0)
        all_labels = np.concatenate(all_labels, 0)
        plot_features(all_features, all_labels, num_classes, epoch, prefix='train')

def test(model, testloader, use_gpu, num_classes, epoch):
    model.eval()
    correct, total = 0, 0
    if args.plot:
        all_features, all_labels = [], []

    with torch.no_grad():
        for index, (data, labels) in testloader:
            if use_gpu:
                data, labels = data.cuda(), labels.cuda()
            features, outputs = model(data)
            predictions = outputs.data.max(1)[1]
            total += labels.size(0)
            correct += (predictions == labels.data).sum()
            
            if args.plot:
                if use_gpu:
                    all_features.append(features.data.cpu().numpy())
                    all_labels.append(labels.data.cpu().numpy())
                else:
                    all_features.append(features.data.numpy())
                    all_labels.append(labels.data.numpy())

    if args.plot:
        all_features = np.concatenate(all_features, 0)
        all_labels = np.concatenate(all_labels, 0)
        plot_features(all_features, all_labels, num_classes, epoch, prefix='test')

    acc = correct * 100. / total
    err = 100. - acc
    return acc, err

def plot_features(features, labels, num_classes, epoch, prefix):
    """Plot features on 2D plane.

    Args:
        features: (num_instances, num_features).
        labels: (num_instances). 
    """
    colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']
    for label_idx in range(num_classes):
        plt.scatter(
            features[labels==label_idx, 0],
            features[labels==label_idx, 1],
            c=colors[label_idx],
            s=1,
        )
    plt.legend(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'], loc='upper right')
    dirname = osp.join(args.save_dir, prefix)
    if not osp.exists(dirname):
        os.mkdir(dirname)
    save_name = osp.join(dirname, 'epoch_' + str(epoch+1) + '.png')
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()





