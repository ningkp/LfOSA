import matplotlib.pyplot as plt
import numpy as np
import pickle

def smoothing(arr):
    newArr = []
    smoothing_rate = 0.95

def diff_temperature_cifar100():
    known = 20
    init = 8
    model = "resnet18"
    seeds = [1]
    max_min_temperature_acc = []
    max_min_temperature_precision = []
    max_min_temperature_recall = []
    min_max_temperature_acc = []
    min_max_temperature_precision = []
    min_max_temperature_recall = []
    max_max_temperature_acc = []
    max_max_temperature_precision = []
    max_max_temperature_recall = []
    min_min_temperature_acc = []
    min_min_temperature_precision = []
    min_min_temperature_recall = []
    min_min_temperature_acc2 = []
    min_min_temperature_precision2 = []
    min_min_temperature_recall2 = []
    mid_mid_temperature_acc = []
    mid_mid_temperature_precision = []
    mid_mid_temperature_recall = []
    min_min_modelB_temperature_acc = []
    min_min_modelB_temperature_precision = []
    min_min_modelB_temperature_recall = []
    min_min_modelB_temperature_acc2 = []
    min_min_modelB_temperature_precision2 = []
    min_min_modelB_temperature_recall2 = []

    for seed in seeds:
        with open("log_AL/temperature_"+model+"_cifar100_known"+str(known)+"_init"+str(init)+"_batch1500_seed2_AV_temperature_unknown_T2.0_known_T0.5.pkl", 'rb') as f:
            data = pickle.load(f)
            max_min_temperature_acc.append([data['Acc'][i] for i in data['Acc']])
            max_min_temperature_precision.append([data['Precision'][i] for i in data['Precision']])
            max_min_temperature_recall.append([data['Recall'][i] for i in data['Recall']])
        f.close()

        with open("log_AL/temperature_"+model+"_cifar100_known"+str(known)+"_init"+str(init)+"_batch1500_seed1_AV_temperature_unknown_T0.5_known_T0.5_modelB_T1.0.pkl", 'rb') as f:
            data = pickle.load(f)
            min_min_temperature_acc.append([data['Acc'][i] for i in data['Acc']])
            min_min_temperature_precision.append([data['Precision'][i] for i in data['Precision']])
            min_min_temperature_recall.append([data['Recall'][i] for i in data['Recall']])
        f.close()

        with open("log_AL/temperature_"+model+"_cifar100_known"+str(known)+"_init"+str(init)+"_batch1500_seed1_AV_temperature_unknown_T0.2_known_T0.2.pkl", 'rb') as f:
            data = pickle.load(f)
            min_min_temperature_acc2.append([data['Acc'][i] for i in data['Acc']])
            min_min_temperature_precision2.append([data['Precision'][i] for i in data['Precision']])
            min_min_temperature_recall2.append([data['Recall'][i] for i in data['Recall']])
        f.close()

        with open("log_AL/temperature_"+model+"_cifar100_known"+str(known)+"_init"+str(init)+"_batch1500_seed1_AV_temperature_unknown_T1.0_known_T1.0.pkl", 'rb') as f:
            data = pickle.load(f)
            mid_mid_temperature_acc.append([data['Acc'][i] for i in data['Acc']])
            mid_mid_temperature_precision.append([data['Precision'][i] for i in data['Precision']])
            mid_mid_temperature_recall.append([data['Recall'][i] for i in data['Recall']])
        f.close()

        with open("log_AL/temperature_"+model+"_cifar100_known"+str(known)+"_init"+str(init)+"_batch1500_seed1_AV_temperature_unknown_T2.0_known_T2.0.pkl", 'rb') as f:
            data = pickle.load(f)
            max_max_temperature_acc.append([data['Acc'][i] for i in data['Acc']])
            max_max_temperature_precision.append([data['Precision'][i] for i in data['Precision']])
            max_max_temperature_recall.append([data['Recall'][i] for i in data['Recall']])
        f.close()

        with open("log_AL/temperature_"+model+"_cifar100_known"+str(known)+"_init"+str(init)+"_batch1500_seed1_AV_temperature_unknown_T0.5_known_T2.0.pkl", 'rb') as f:
            data = pickle.load(f)
            min_max_temperature_acc.append([data['Acc'][i] for i in data['Acc']])
            min_max_temperature_precision.append([data['Precision'][i] for i in data['Precision']])
            min_max_temperature_recall.append([data['Recall'][i] for i in data['Recall']])
        f.close()

        with open("log_AL/temperature_"+model+"_cifar100_known"+str(known)+"_init"+str(init)+"_batch1500_seed1_AV_temperature_unknown_T0.5_known_T0.5_modelB_T1.2.pkl", 'rb') as f:
            data = pickle.load(f)
            min_min_modelB_temperature_acc.append([data['Acc'][i] for i in data['Acc']])
            min_min_modelB_temperature_precision.append([data['Precision'][i] for i in data['Precision']])
            min_min_modelB_temperature_recall.append([data['Recall'][i] for i in data['Recall']])
        f.close()

        with open("log_AL/temperature_"+model+"_cifar100_known"+str(known)+"_init"+str(init)+"_batch1500_seed1_AV_temperature_unknown_T0.5_known_T0.5_modelB_T1.5.pkl", 'rb') as f:
            data = pickle.load(f)
            min_min_modelB_temperature_acc2.append([data['Acc'][i] for i in data['Acc']])
            min_min_modelB_temperature_precision2.append([data['Precision'][i] for i in data['Precision']])
            min_min_modelB_temperature_recall2.append([data['Recall'][i] for i in data['Recall']])
        f.close()

    x = list(range(10))
    plt.figure()
    plt.title("Recall")
    plt.plot(x, np.array(max_min_temperature_recall).mean(0), label='unknown2.0_known0.5')
    plt.plot(x, np.array(min_min_temperature_recall).mean(0), label='unknown0.5_known0.5')
    plt.plot(x, np.array(min_min_modelB_temperature_recall).mean(0), label='unknown0.5_known0.5_modelB1.2')
    plt.plot(x, np.array(min_min_modelB_temperature_recall2).mean(0), label='unknown0.5_known0.5_modelB1.5')
    plt.plot(x, np.array(min_min_temperature_recall2).mean(0), label='unknown0.2_known0.2')
    plt.plot(x, np.array(mid_mid_temperature_recall).mean(0), label='unknown1.0_known1.0')
    plt.plot(x, np.array(max_max_temperature_recall).mean(0), label='unknown2.0_known2.0')
    plt.plot(x, np.array(min_max_temperature_recall).mean(0), label='unknown0.5_known2.0')

    plt.fill_between(x, np.array(max_min_temperature_recall).mean(0) - np.array(max_min_temperature_recall).std(0),
                     np.array(max_min_temperature_recall).mean(0) + np.array(max_min_temperature_recall).std(0),
                     color='r',
                     alpha=0.2)
    plt.fill_between(x, np.array(min_min_temperature_recall).mean(0) - np.array(min_min_temperature_recall).std(0),
                     np.array(min_min_temperature_recall).mean(0) + np.array(min_min_temperature_recall).std(0),
                     color='b',
                     alpha=0.2)
    plt.fill_between(x, np.array(min_min_temperature_recall2).mean(0) - np.array(min_min_temperature_recall2).std(0),
                     np.array(min_min_temperature_recall2).mean(0) + np.array(min_min_temperature_recall2).std(0),
                     color='m',
                     alpha=0.2)
    plt.fill_between(x, np.array(mid_mid_temperature_recall).mean(0) - np.array(mid_mid_temperature_recall).std(0),
                     np.array(mid_mid_temperature_recall).mean(0) + np.array(mid_mid_temperature_recall).std(0),
                     color='k',
                     alpha=0.2)
    plt.fill_between(x, np.array(max_max_temperature_recall).mean(0) - np.array(max_max_temperature_recall).std(0),
                     np.array(max_max_temperature_recall).mean(0) + np.array(max_max_temperature_recall).std(0),
                     color='g',
                     alpha=0.2)
    plt.fill_between(x, np.array(min_max_temperature_recall).mean(0) - np.array(min_max_temperature_recall).std(0),
                     np.array(min_max_temperature_recall).mean(0) + np.array(min_max_temperature_recall).std(0),
                     color='y',
                     alpha=0.2)
    plt.legend(loc='best')
    plt.savefig("gifs/cifar100_diff_temperature_"+model+"_init"+str(init)+"_known"+str(known)+"_recall.png")
    plt.show()

    plt.figure()
    plt.title("Precision")
    plt.plot(x, np.array(max_min_temperature_precision).mean(0), label='unknown2.0_known0.5')
    plt.plot(x, np.array(min_min_temperature_precision).mean(0), label='unknown0.5_known0.5')
    plt.plot(x, np.array(min_min_modelB_temperature_precision).mean(0), label='unknown0.5_known0.5_modelB1.2')
    plt.plot(x, np.array(min_min_modelB_temperature_precision2).mean(0), label='unknown0.5_known0.5_modelB1.5')
    plt.plot(x, np.array(min_min_temperature_precision2).mean(0), label='unknown0.2_known0.2')
    plt.plot(x, np.array(mid_mid_temperature_precision).mean(0), label='unknown1.0_known1.0')
    plt.plot(x, np.array(max_max_temperature_precision).mean(0), label='unknown2.0_known2.0')
    plt.plot(x, np.array(min_max_temperature_precision).mean(0), label='unknown0.5_known2.0')

    plt.fill_between(x, np.array(max_min_temperature_precision).mean(0) - np.array(max_min_temperature_precision).std(0),
                     np.array(max_min_temperature_precision).mean(0) + np.array(max_min_temperature_precision).std(0),
                     color='r',
                     alpha=0.2)
    plt.fill_between(x, np.array(min_min_temperature_precision).mean(0) - np.array(min_min_temperature_precision).std(0),
                     np.array(min_min_temperature_precision).mean(0) + np.array(min_min_temperature_precision).std(0),
                     color='b',
                     alpha=0.2)
    plt.fill_between(x, np.array(min_min_temperature_precision2).mean(0) - np.array(min_min_temperature_precision2).std(0),
                     np.array(min_min_temperature_precision2).mean(0) + np.array(min_min_temperature_precision2).std(0),
                     color='m',
                     alpha=0.2)
    plt.fill_between(x, np.array(mid_mid_temperature_precision).mean(0) - np.array(mid_mid_temperature_precision).std(0),
                     np.array(mid_mid_temperature_precision).mean(0) + np.array(mid_mid_temperature_precision).std(0),
                     color='k',
                     alpha=0.2)
    plt.fill_between(x, np.array(max_max_temperature_precision).mean(0) - np.array(max_max_temperature_precision).std(0),
                     np.array(max_max_temperature_precision).mean(0) + np.array(max_max_temperature_precision).std(0),
                     color='g',
                     alpha=0.2)
    plt.fill_between(x, np.array(min_max_temperature_precision).mean(0) - np.array(min_max_temperature_precision).std(0),
                     np.array(min_max_temperature_precision).mean(0) + np.array(min_max_temperature_precision).std(0),
                     color='y',
                     alpha=0.2)
    plt.legend(loc='best')
    plt.savefig("gifs/cifar100_diff_temperature_"+model+"_init"+str(init)+"_known"+str(known)+"_precision.png")
    plt.show()

    plt.figure()
    plt.title("Acc")
    plt.plot(x, np.array(max_min_temperature_acc).mean(0), label='unknown2.0_known0.5')
    plt.plot(x, np.array(min_min_temperature_acc).mean(0), label='unknown0.5_known0.5')
    plt.plot(x, np.array(min_min_modelB_temperature_acc).mean(0), label='unknown0.5_known0.5_modelB1.2')
    plt.plot(x, np.array(min_min_modelB_temperature_acc2).mean(0), label='unknown0.5_known0.5_modelB1.5')
    plt.plot(x, np.array(min_min_temperature_acc2).mean(0), label='unknown0.2_known0.2')
    plt.plot(x, np.array(mid_mid_temperature_acc).mean(0), label='unknown1.0_known1.0')
    plt.plot(x, np.array(max_max_temperature_acc).mean(0), label='unknown2.0_known2.0')
    plt.plot(x, np.array(min_max_temperature_acc).mean(0), label='unknown0.5_known2.0')

    plt.fill_between(x, np.array(max_min_temperature_acc).mean(0) - np.array(max_min_temperature_acc).std(0),
                     np.array(max_min_temperature_acc).mean(0) + np.array(max_min_temperature_acc).std(0),
                     color='r',
                     alpha=0.2)
    plt.fill_between(x, np.array(min_min_temperature_acc).mean(0) - np.array(min_min_temperature_acc).std(0),
                     np.array(min_min_temperature_acc).mean(0) + np.array(min_min_temperature_acc).std(0),
                     color='b',
                     alpha=0.2)
    plt.fill_between(x, np.array(min_min_temperature_acc2).mean(0) - np.array(min_min_temperature_acc2).std(0),
                     np.array(min_min_temperature_acc2).mean(0) + np.array(min_min_temperature_acc2).std(0),
                     color='m',
                     alpha=0.2)
    plt.fill_between(x, np.array(mid_mid_temperature_acc).mean(0) - np.array(mid_mid_temperature_acc).std(0),
                     np.array(mid_mid_temperature_acc).mean(0) + np.array(mid_mid_temperature_acc).std(0),
                     color='k',
                     alpha=0.2)
    plt.fill_between(x, np.array(max_max_temperature_acc).mean(0) - np.array(max_max_temperature_acc).std(0),
                     np.array(max_max_temperature_acc).mean(0) + np.array(max_max_temperature_acc).std(0),
                     color='g',
                     alpha=0.2)
    plt.fill_between(x, np.array(min_max_temperature_acc).mean(0) - np.array(min_max_temperature_acc).std(0),
                     np.array(min_max_temperature_acc).mean(0) + np.array(min_max_temperature_acc).std(0),
                     color='y',
                     alpha=0.2)
    plt.legend(loc='best')
    plt.savefig("gifs/cifar100_diff_temperature_"+model+"_init"+str(init)+"_known"+str(known)+"_accuracy.png")
    plt.show()

def plot_performance_cifar100():
    known = 20
    init = 8
    model = "resnet18"
    seeds = [1, 2, 3, 4]
    random_acc = []
    uncertainty_acc = []
    AV_based_acc = []
    max_av_acc = []
    temperature_acc = []
    random_precision = []
    uncertainty_precision = []
    AV_based_precision = []
    max_av_precision = []
    temperature_precision = []
    random_recall = []
    uncertainty_recall = []
    AV_based_recall = []
    max_av_recall = []
    temperature_recall = []

    for seed in seeds:
        with open("log_AL/"+model+"_cifar100_known"+str(known)+"_init"+str(init)+"_batch1500_seed"+str(seed)+"_random.pkl", 'rb') as f:
            data = pickle.load(f)
            random_acc.append([data['Acc'][i] for i in data['Acc']])
            random_precision.append([data['Precision'][i] for i in data['Precision']])
            random_recall.append([data['Recall'][i] for i in data['Recall']])
        f.close()
        with open("log_AL/"+model+"_cifar100_known"+str(known)+"_init"+str(init)+"_batch1500_seed"+str(seed)+"_uncertainty.pkl", 'rb') as f:
            data = pickle.load(f)
            uncertainty_acc.append([data['Acc'][i] for i in data['Acc']])
            uncertainty_precision.append([data['Precision'][i] for i in data['Precision']])
            uncertainty_recall.append([data['Recall'][i] for i in data['Recall']])
        f.close()
        with open("log_AL/"+model+"_cifar100_known"+str(known)+"_init"+str(init)+"_batch1500_seed"+str(seed)+"_AV_based.pkl", 'rb') as f:
            data = pickle.load(f)
            AV_based_acc.append([data['Acc'][i] for i in data['Acc']])
            AV_based_precision.append([data['Precision'][i] for i in data['Precision']])
            AV_based_recall.append([data['Recall'][i] for i in data['Recall']])
        f.close()
        with open("log_AL/"+model+"_cifar100_known"+str(known)+"_init"+str(init)+"_batch1500_seed"+str(seed)+"_AV_uncertainty.pkl", 'rb') as f:
            data = pickle.load(f)
            max_av_acc.append([data['Acc'][i] for i in data['Acc']])
            max_av_precision.append([data['Precision'][i] for i in data['Precision']])
            max_av_recall.append([data['Recall'][i] for i in data['Recall']])
        f.close()
        with open("log_AL/temperature_"+model+"_cifar100_known"+str(known)+"_init"+str(init)+"_batch1500_seed"+str(seed)+"_AV_temperature_unknown_T0.5_known_T0.5_modelB_T1.0.pkl", 'rb') as f:
            data = pickle.load(f)
            temperature_acc.append([data['Acc'][i] for i in data['Acc']])
            temperature_precision.append([data['Precision'][i] for i in data['Precision']])
            temperature_recall.append([data['Recall'][i] for i in data['Recall']])
        f.close()
    x = list(range(10))
    plt.figure()
    plt.title("Recall")
    plt.plot(x, np.array(random_recall).mean(0), label='random')
    plt.plot(x, np.array(uncertainty_recall).mean(0), label='uncertainty')
    plt.plot(x, np.array(AV_based_recall).mean(0), label='AV_based')
    plt.plot(x, np.array(max_av_recall).mean(0), label='AV_uncertainty')
    plt.plot(x, np.array(temperature_recall).mean(0), label='AV_temperature_framework')

    plt.fill_between(x, np.array(random_recall).mean(0) - np.array(random_recall).std(0),
                     np.array(random_recall).mean(0) + np.array(random_recall).std(0),
                     color='b',
                     alpha=0.2)
    plt.fill_between(x, np.array(uncertainty_recall).mean(0) - np.array(uncertainty_recall).std(0),
                     np.array(uncertainty_recall).mean(0) + np.array(uncertainty_recall).std(0),
                     color='r',
                     alpha=0.2)
    plt.fill_between(x, np.array(AV_based_recall).mean(0) - np.array(AV_based_recall).std(0),
                     np.array(AV_based_recall).mean(0) + np.array(AV_based_recall).std(0),
                     color='g',
                     alpha=0.2)
    plt.fill_between(x, np.array(max_av_recall).mean(0) - np.array(max_av_recall).std(0),
                     np.array(max_av_recall).mean(0) + np.array(max_av_recall).std(0),
                     color='y',
                     alpha=0.2)
    plt.fill_between(x, np.array(temperature_recall).mean(0) - np.array(temperature_recall).std(0),
                     np.array(temperature_recall).mean(0) + np.array(temperature_recall).std(0),
                     color='k',
                     alpha=0.2)
    plt.legend(loc='best')
    plt.savefig("gifs/cifar100_"+model+"_init"+str(init)+"_known"+str(known)+"_recall.png")
    plt.show()

    plt.figure()
    plt.title("Precision")
    plt.plot(x, np.array(random_precision).mean(0), label='random')
    plt.plot(x, np.array(uncertainty_precision).mean(0), label='uncertainty')
    plt.plot(x, np.array(AV_based_precision).mean(0), label='AV_based')
    plt.plot(x, np.array(max_av_precision).mean(0), label='AV_uncertainty')
    plt.plot(x, np.array(temperature_precision).mean(0), label='AV_temperature_framework')
    plt.fill_between(x, np.array(random_precision).mean(0) - np.array(random_precision).std(0),
                     np.array(random_precision).mean(0) + np.array(random_precision).std(0),
                     color='b',
                     alpha=0.2)
    plt.fill_between(x, np.array(uncertainty_precision).mean(0) - np.array(uncertainty_precision).std(0),
                     np.array(uncertainty_precision).mean(0) + np.array(uncertainty_precision).std(0),
                     color='r',
                     alpha=0.2)
    plt.fill_between(x, np.array(AV_based_precision).mean(0) - np.array(AV_based_precision).std(0),
                     np.array(AV_based_precision).mean(0) + np.array(AV_based_precision).std(0),
                     color='g',
                     alpha=0.2)
    plt.fill_between(x, np.array(max_av_precision).mean(0) - np.array(max_av_precision).std(0),
                     np.array(max_av_precision).mean(0) + np.array(max_av_precision).std(0),
                     color='y',
                     alpha=0.2)
    plt.fill_between(x, np.array(temperature_precision).mean(0) - np.array(temperature_precision).std(0),
                     np.array(temperature_precision).mean(0) + np.array(temperature_precision).std(0),
                     color='k',
                     alpha=0.2)
    plt.legend(loc='best')
    plt.savefig("gifs/cifar100_"+model+"_init"+str(init)+"_known"+str(known)+"_precision.png")
    plt.show()

    plt.figure()
    plt.title("Acc")
    plt.plot(x, np.array(random_acc).mean(0), label='random')
    plt.plot(x, np.array(uncertainty_acc).mean(0), label='uncertainty')
    plt.plot(x, np.array(AV_based_acc).mean(0), label='AV_based')
    plt.plot(x, np.array(max_av_acc).mean(0), label='AV_uncertainty')
    plt.plot(x, np.array(temperature_acc).mean(0), label='AV_temperature_framework')
    plt.fill_between(x, np.array(random_acc).mean(0) - np.array(random_acc).std(0),
                     np.array(random_acc).mean(0) + np.array(random_acc).std(0),
                     color='b',
                     alpha=0.2)
    plt.fill_between(x, np.array(uncertainty_acc).mean(0) - np.array(uncertainty_acc).std(0),
                     np.array(uncertainty_acc).mean(0) + np.array(uncertainty_acc).std(0),
                     color='r',
                     alpha=0.2)
    plt.fill_between(x, np.array(AV_based_acc).mean(0) - np.array(AV_based_acc).std(0),
                     np.array(AV_based_acc).mean(0) + np.array(AV_based_acc).std(0),
                     color='g',
                     alpha=0.2)
    plt.fill_between(x, np.array(max_av_acc).mean(0) - np.array(max_av_acc).std(0),
                     np.array(max_av_acc).mean(0) + np.array(max_av_acc).std(0),
                     color='y',
                     alpha=0.2)
    plt.fill_between(x, np.array(temperature_acc).mean(0) - np.array(temperature_acc).std(0),
                     np.array(temperature_acc).mean(0) + np.array(temperature_acc).std(0),
                     color='k',
                     alpha=0.2)
    plt.legend(loc='best')
    plt.savefig("gifs/cifar100_"+model+"_init"+str(init)+"_known"+str(known)+"_accuracy.png")
    plt.show()

def plot_performance_cifar10():
    known = 8
    init = 1
    model = "resnet18"
    seeds = [1, 2, 3, 4]
    random_acc = []
    uncertainty_acc = []
    AV_based_acc = []
    max_av_acc = []
    temperature_acc = []
    random_precision = []
    uncertainty_precision = []
    AV_based_precision = []
    max_av_precision = []
    temperature_precision = []
    random_recall = []
    uncertainty_recall = []
    AV_based_recall = []
    max_av_recall = []
    temperature_recall = []

    for seed in seeds:
        with open("log_AL/"+model+"_cifar10_known"+str(known)+"_init"+str(init)+"_batch1500_seed"+str(seed)+"_random.pkl", 'rb') as f:
            data = pickle.load(f)
            random_acc.append([data['Acc'][i] for i in data['Acc']])
            random_precision.append([data['Precision'][i] for i in data['Precision']])
            random_recall.append([data['Recall'][i] for i in data['Recall']])
        f.close()

        with open("log_AL/"+model+"_cifar10_known"+str(known)+"_init"+str(init)+"_batch1500_seed"+str(seed)+"_uncertainty.pkl", 'rb') as f:
            data = pickle.load(f)
            uncertainty_acc.append([data['Acc'][i] for i in data['Acc']])
            uncertainty_precision.append([data['Precision'][i] for i in data['Precision']])
            uncertainty_recall.append([data['Recall'][i] for i in data['Recall']])
        f.close()

        with open("log_AL/temperature_"+model+"_cifar10_known"+str(known)+"_init"+str(init)+"_batch1500_seed"+str(seed)+"_AV_temperature_unknown_T0.5_known_T0.5_modelB_T1.0.pkl", 'rb') as f:
            data = pickle.load(f)
            temperature_acc.append([data['Acc'][i] for i in data['Acc']])
            temperature_precision.append([data['Precision'][i] for i in data['Precision']])
            temperature_recall.append([data['Recall'][i] for i in data['Recall']])
        f.close()

        with open("log_AL/"+model+"_cifar10_known"+str(known)+"_init"+str(init)+"_batch1500_seed"+str(seed)+"_AV_based.pkl", 'rb') as f:
            data = pickle.load(f)
            AV_based_acc.append([data['Acc'][i] for i in data['Acc']])
            AV_based_precision.append([data['Precision'][i] for i in data['Precision']])
            AV_based_recall.append([data['Recall'][i] for i in data['Recall']])
        f.close()

    x = list(range(10))
    plt.figure()
    plt.title("Recall")
    plt.plot(x, np.array(random_recall).mean(0), label='random')
    plt.plot(x, np.array(AV_based_recall).mean(0), label='AV_based')
    plt.plot(x, np.array(uncertainty_recall).mean(0), label='uncertainty')
    plt.plot(x, np.array(temperature_recall).mean(0), label='AV_temperature_framework')

    plt.fill_between(x, np.array(random_recall).mean(0) - np.array(random_recall).std(0),
                     np.array(random_recall).mean(0) + np.array(random_recall).std(0),
                     color='b',
                     alpha=0.2)
    plt.fill_between(x, np.array(AV_based_recall).mean(0) - np.array(AV_based_recall).std(0),
                     np.array(AV_based_recall).mean(0) + np.array(AV_based_recall).std(0),
                     color='g',
                     alpha=0.2)
    plt.fill_between(x, np.array(uncertainty_recall).mean(0) - np.array(uncertainty_recall).std(0),
                     np.array(uncertainty_recall).mean(0) + np.array(uncertainty_recall).std(0),
                     color='r',
                     alpha=0.2)
    plt.fill_between(x, np.array(temperature_recall).mean(0) - np.array(temperature_recall).std(0),
                     np.array(temperature_recall).mean(0) + np.array(temperature_recall).std(0),
                     color='k',
                     alpha=0.2)
    plt.legend(loc='best')
    plt.savefig("gifs/cifar10_"+model+"_init"+str(init)+"_known"+str(known)+"_recall.png")
    plt.show()

    plt.figure()
    plt.title("Precision")
    plt.plot(x, np.array(random_precision).mean(0), label='random')
    plt.plot(x, np.array(AV_based_precision).mean(0), label='AV_based')
    plt.plot(x, np.array(uncertainty_precision).mean(0), label='uncertainty')
    plt.plot(x, np.array(temperature_precision).mean(0), label='AV_temperature_framework')
    plt.fill_between(x, np.array(random_precision).mean(0) - np.array(random_precision).std(0),
                     np.array(random_precision).mean(0) + np.array(random_precision).std(0),
                     color='b',
                     alpha=0.2)
    plt.fill_between(x, np.array(AV_based_precision).mean(0) - np.array(AV_based_precision).std(0),
                     np.array(AV_based_precision).mean(0) + np.array(AV_based_precision).std(0),
                     color='g',
                     alpha=0.2)
    plt.fill_between(x, np.array(uncertainty_precision).mean(0) - np.array(uncertainty_precision).std(0),
                     np.array(uncertainty_precision).mean(0) + np.array(uncertainty_precision).std(0),
                     color='r',
                     alpha=0.2)
    plt.fill_between(x, np.array(temperature_precision).mean(0) - np.array(temperature_precision).std(0),
                     np.array(temperature_precision).mean(0) + np.array(temperature_precision).std(0),
                     color='k',
                     alpha=0.2)
    plt.legend(loc='best')
    plt.savefig("gifs/cifar10_"+model+"_init"+str(init)+"_known"+str(known)+"_precision.png")
    plt.show()

    plt.figure()
    plt.title("Acc")
    plt.plot(x, np.array(random_acc).mean(0), label='random')
    plt.plot(x, np.array(AV_based_acc).mean(0), label='AV_based')
    plt.plot(x, np.array(uncertainty_acc).mean(0), label='uncertainty')
    plt.plot(x, np.array(temperature_acc).mean(0), label='AV_temperature_framework')
    plt.fill_between(x, np.array(random_acc).mean(0) - np.array(random_acc).std(0),
                     np.array(random_acc).mean(0) + np.array(random_acc).std(0),
                     color='b',
                     alpha=0.2)
    plt.fill_between(x, np.array(AV_based_acc).mean(0) - np.array(AV_based_acc).std(0),
                     np.array(AV_based_acc).mean(0) + np.array(AV_based_acc).std(0),
                     color='g',
                     alpha=0.2)
    plt.fill_between(x, np.array(uncertainty_acc).mean(0) - np.array(uncertainty_acc).std(0),
                     np.array(uncertainty_acc).mean(0) + np.array(uncertainty_acc).std(0),
                     color='r',
                     alpha=0.2)
    plt.fill_between(x, np.array(temperature_acc).mean(0) - np.array(temperature_acc).std(0),
                     np.array(temperature_acc).mean(0) + np.array(temperature_acc).std(0),
                     color='k',
                     alpha=0.2)
    plt.legend(loc='best')
    plt.savefig("gifs/cifar10_"+model+"_init"+str(init)+"_known"+str(known)+"_accuracy.png")
    plt.show()

def plot_distribution():
    with open("pkl/center_result.pkl", 'rb') as f:
        data = pickle.load(f)
    f.close()

    known_S_ij = data['known_S']
    unknown_S_ij = data['unknown_S']
    known_M_ij = data['known_M']
    unknown_M_ij = data['unknown_M']

    for i in range(7):
        # i = 23
        known_data = known_S_ij[i]
        unknown_data = unknown_S_ij[i]
        plt.hist(known_data, bins=40, color="#FF0000", alpha=.9)
        plt.hist(unknown_data, bins=40, color="#C1F320", alpha=.5)
        plt.savefig("log/AV_CentorLoss_result_mini/class_"+str(i)+".png")
        plt.show()
        # print(ca)

plot_performance_cifar100()
plot_performance_cifar10()