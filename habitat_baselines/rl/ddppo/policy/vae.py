import torch
import torch.nn as nn

""" Configs """
in_channels = 4
base_planes = 32
stride = 1
groups = 1
ngroups = 1
latent_dim = 32
original_dim = 128
hidden_planes = [16, 8, 4]

class Encoder(nn.Module):
    
    def __init__(self, in_channels, base_planes, latent_dim):
        hidden_planes = [16, 8, 4]
        hidden_dim = hidden_planes[2] * hidden_planes[1] * hidden_planes[1]

        super(Encoder, self).__init__()
        self.FC_input = nn.Sequential(
            nn.Conv2d(
                in_channels,
                base_planes,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            ),
            nn.BatchNorm2d(base_planes),
            nn.Conv2d(
                  base_planes,
                  hidden_planes[0],
                  kernel_size=7,
                  stride=2,
                  padding=3,
                  bias=False,
              ),
            nn.BatchNorm2d(hidden_planes[0]),
            nn.Conv2d(
                  hidden_planes[0],
                  hidden_planes[1],
                  kernel_size=7,
                  stride=2,
                  padding=3,
                  bias=False,
              ),
            nn.BatchNorm2d(hidden_planes[1]), 
            nn.Conv2d(
                  hidden_planes[1],
                  hidden_planes[2],
                  kernel_size=7,
                  stride=2,
                  padding=3,
                  bias=False,
              ),
            nn.BatchNorm2d(hidden_planes[2]),
            nn.Flatten()
        )
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim, latent_dim)
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
        
        self.training = True
        
    def forward(self, x):
        h_       = self.LeakyReLU(self.FC_input(x))
        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)                     # encoder produces mean and log of variance 
                                                       #             (i.e., parateters of simple tractable normal distribution "q"
        
        return mean, log_var

class Decoder(nn.Module):
    def __init__(self, latent_dim, in_channels, base_planes, original_dim, hidden_planes):
        super(Decoder, self).__init__()
        self.hidden_planes = hidden_planes
        hidden_dim = self.hidden_planes[2] * self.hidden_planes[1] * self.hidden_planes[1]

        self.roll = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden = nn.Sequential(
            nn.ConvTranspose2d(self.hidden_planes[2], self.hidden_planes[1], self.hidden_planes[1]+2, 2),
            nn.BatchNorm2d(self.hidden_planes[1]),
            nn.ConvTranspose2d(self.hidden_planes[1], base_planes, self.hidden_planes[1]+1),
            nn.BatchNorm2d(base_planes),
        )
        self.FC_hidden2 = nn.Sequential(
            nn.ConvTranspose2d(base_planes, self.hidden_planes[0], self.hidden_planes[0] * 2 + 1),
            nn.BatchNorm2d(self.hidden_planes[0]),
            nn.ConvTranspose2d(self.hidden_planes[0], in_channels, original_dim//2+1),
            nn.BatchNorm2d(in_channels),
        )
        
        self.LeakyReLU = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        h = self.roll(x)
        h = h.view((-1, self.hidden_planes[2], self.hidden_planes[1], self.hidden_planes[1]))
        x_latent = self.FC_hidden(h)
        x_hat = self.FC_hidden2(x_latent)

        return x_latent, x_hat

class VAE(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(VAE, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
        self.mean = 128 * 128 * 4
        
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var)       # sampling epsilon        
        z = mean + var*epsilon                          # reparameterization trick
        return z
        
                
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_latent, x_hat            = self.Decoder(z)
        reproduction_loss = nn.functional.mse_loss(x_hat, x, reduction='sum') / self.mean
        KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
        loss = reproduction_loss + KLD
        return x_latent, loss


def create_vae_model( in_channels, base_planes, original_dim=128, hidden_planes=hidden_planes, latent_dim=latent_dim,):
    encoder = Encoder(in_channels, base_planes, latent_dim)
    decoder = Decoder(latent_dim, in_channels, base_planes, original_dim, hidden_planes)

    return VAE(encoder, decoder)

