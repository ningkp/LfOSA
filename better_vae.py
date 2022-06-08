import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torchvision
from torchvision import transforms
import torch.optim as optim
from torch import nn
import matplotlib.pyplot as plt
import os
from tqdm import tqdm


class Encoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Encoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.tanh(self.linear2(x))


class Decoder(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(Decoder, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, D_out)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        return F.tanh(self.linear2(x))


class VAE(torch.nn.Module):

    def __init__(self, encoder, decoder, num_classes):
        super(VAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.latent_dim = 32
        self._enc_mu = torch.nn.Linear(100, self.latent_dim)
        self._enc_log_sigma = torch.nn.Linear(100, self.latent_dim)
        self.mu_bn = torch.nn.BatchNorm1d(self.latent_dim)
        self.mu_bn.weight.requires_grad = False
        nn.init.constant_(self.mu_bn.bias, 0.0)
        self.mu_bn.weight.fill_(0.5)

        # classifier
        self.fc1 = torch.nn.Linear(self.latent_dim, 16)
        self.fc2 = torch.nn.Linear(16, num_classes)

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        self.z_mean = self.mu_bn(mu)
        self.z_sigma = sigma

        self.z = self.z_mean + self.z_sigma * Variable(std_z, requires_grad=False).cuda()
        return self.z  # Reparameterization trick

    def forward(self, state):
        h_enc = self.encoder(state)
        z = self._sample_latent(h_enc)
        # classify
        # print(z.shape)
        # print(self.fc1)
        output = self.fc1(z)
        output = self.fc2(output)
        return self.decoder(z), output


class VAE2(torch.nn.Module):

    def __init__(self, enc_out_dim=512, latent_dim=256, input_height=32):
        super(VAE2, self).__init__()
        self.encoder = resnet18_encoder(False, False)
        self.decoder = resnet18_decoder(
            latent_dim=latent_dim,
            input_height=input_height,
            first_conv=False,
            maxpool1=False
        )

        self.latent_dim = latent_dim
        self._enc_mu = torch.nn.Linear(100, self.latent_dim)
        self._enc_log_sigma = torch.nn.Linear(100, self.latent_dim)
        self.mu_bn = torch.nn.BatchNorm1d(self.latent_dim)
        self.mu_bn.weight.requires_grad = False
        nn.init.constant_(self.mu_bn.bias, 0.0)
        self.mu_bn.weight.fill_(0.5)

    def _sample_latent(self, h_enc):
        """
        Return the latent normal sample z ~ N(mu, sigma^2)
        """
        mu = self._enc_mu(h_enc)
        log_sigma = self._enc_log_sigma(h_enc)
        sigma = torch.exp(log_sigma)
        std_z = torch.from_numpy(np.random.normal(0, 1, size=sigma.size())).float()

        self.z_mean = self.mu_bn(mu)
        self.z_sigma = sigma

        self.z = self.z_mean + self.z_sigma * Variable(std_z, requires_grad=False)
        return self.z  # Reparameterization trick

    def forward(self, state):
        h_enc = self.encoder(state)
        z = self._sample_latent(h_enc)
        return self.decoder(z)


def gaussian_likelihood(x_hat, x):
    log_scale = nn.Parameter(torch.Tensor([0.0]).cuda())
    scale = torch.exp(log_scale)
    mean = x_hat
    dist = torch.distributions.Normal(mean, scale)

    # measure prob of seeing image under p(x|z)
    log_pxz = dist.log_prob(x)
    return log_pxz.sum(dim=(1))


def kl_divergence(z, mu, std):
    # --------------------------
    # Monte carlo KL divergence
    # --------------------------
    # 1. define the first two probabilities (in this case Normal for both)
    p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
    q = torch.distributions.Normal(mu, std)

    # 2. get the probabilities from the equation
    log_qzx = q.log_prob(z)
    log_pz = p.log_prob(z)

    # kl
    kl = (log_qzx - log_pz)
    kl = kl.sum(-1)
    return kl


def show_image_grid(images, epoch, batch_size=8, name=""):
    fig = plt.figure(figsize=(8, batch_size / 10))
    # fig.suptitle("Pass {}".format(pass_id))
    gs = plt.GridSpec(int(batch_size / 10) + 1, 10)
    gs.update(wspace=0.05, hspace=0.05)

    mu, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)

    for i, image in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')

        # image = image.float().numpy()
        image = np.array(image).reshape(3, 32, 32)
        for i in range(len(mu)):
            image[i] = image[i] * std[i] + mu[i]
        # image = image * 255
        # print(image.shape)
        img = np.transpose(image, (1, 2, 0))
        plt.imshow(img)

    plt.savefig("log/fig_vae/"+str(epoch)+"_"+name+".png")
    plt.show()


if __name__ == '__main__':

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        print("current gpu used: 0")

    input_dim = 28 * 28
    batch_size = 128

    transform = transforms.Compose([transforms.Resize([28, 28]),
                                    # transforms.Normalize(0.5,0.5),
                                    transforms.ToTensor(),
                                    transforms.Normalize(0.5, 0.5)
                                    ])

    mnist = torchvision.datasets.MNIST('./', download=True, transform=transform)

    dataloader = torch.utils.data.DataLoader(mnist, batch_size=batch_size, shuffle=True, num_workers=2)

    print('Number of samples: ', len(mnist))

    encoder = Encoder(input_dim, 100, 100)
    decoder = Decoder(32, 100, input_dim)
    vae = VAE(encoder, decoder)
    if use_gpu:
        # vae = nn.DataParallel(vae).cuda()
        vae = vae.cuda()

    # criterion = nn.MSELoss(size_average=False)

    optimizer = optim.Adam(vae.parameters(), lr=0.0001)
    l = None
    for epoch in tqdm(range(100)):
        vae.train()
        for i, data in enumerate(dataloader, 0):
            inputs, classes = data

            # inputs, classes = Variable(inputs.resize_(batch_size, input_dim)), Variable(classes)
            if use_gpu:
                inputs, classes = inputs.cuda(), classes.cuda()
            optimizer.zero_grad()
            dec = vae(inputs.view(-1, 784))
            # ll = latent_loss(vae.z_mean, vae.z_sigma)
            ll = kl_divergence(vae.z, vae.z_mean, vae.z_sigma)  # 散度，尽可能小
            # dec_ll = criterion(dec, inputs)
            dec_ll = gaussian_likelihood(dec, inputs.view(-1, 784)) # 重构（似然），尽可能大

            elbo = (ll - dec_ll)

            loss = elbo.mean()

            loss.backward()
            optimizer.step()
        vae.eval()
        dec = vae(inputs.view(-1, 784))
        out = dec.cpu().detach().numpy()
        show_image_grid(inputs.cpu(), epoch, batch_size, "input")
        print(epoch, dec_ll.mean().item(), ll.mean().item(), loss.item())

        show_image_grid(out, epoch, batch_size, "out")

    plt.imshow(vae(inputs).item().numpy().reshape(28, 28), cmap='gray')
    plt.show(block=True)

