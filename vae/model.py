import torch
import torch.nn as nn
import torch.nn.functional as F

'''Simple VAE code'''

class ResBlock(nn.Module):
    def __init__(self, c_num):
        super(ResBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(c_num, c_num, 3, padding=1),
            nn.ReLU(),
            nn.InstanceNorm2d(c_num),
            nn.Conv2d(c_num, c_num, 3, padding=1),
            nn.ReLU(),
            nn.InstanceNorm2d(c_num)
        )   
    
    def forward(self, x):
        out = self.layer(x)
        out = x + out
        return out

class VAE(nn.Module):
    def __init__(self, img_size=28**2, h_dim=400, z_dim=50):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.enc = nn.Sequential(
            nn.Linear(img_size, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, z_dim*2)
        )
        self.dec = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, img_size)
        )
        
    def encode(self, x):
        pre_z = self.enc(x)
        mu = pre_z[:, :self.z_dim]
        log_var = pre_z[:, self.z_dim:]
        return mu, log_var
    
    def reparameterize(self, mu, log_var):
        sigma = torch.exp(log_var/2)
        return mu + sigma * torch.randn(sigma.size()).cuda()
    
    def decode(self, z):
        return F.sigmoid(self.dec(z))
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var

class ConvVAE(nn.Module):
    def __init__(self, img_size=(28, 28), c_num=1, h_dim=3000, z_dim=400):
        super(ConvVAE, self).__init__()
        self.z_dim = z_dim
        self.img_h = img_size[0]
        self.enc_conv = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(c_num, 32, 3, padding=1),
                nn.ReLU(),
                nn.InstanceNorm2d(32)
            ),
            ResBlock(32),
            ResBlock(32),
            nn.MaxPool2d(2),
            nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.InstanceNorm2d(64)
            ),
            ResBlock(64),
            ResBlock(64),
            nn.MaxPool2d(2)
        )
        self.enc_linear = nn.Sequential(
            nn.Sequential(
                nn.Dropout(),
                nn.Linear((self.img_h**2)*64//16, h_dim),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Dropout(),
                nn.Linear(h_dim, 2*z_dim), # regularization only on linear
            ),
        )
        
        self.dec_linear = nn.Sequential(
            nn.Linear(z_dim, h_dim),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(h_dim, (self.img_h**2)*64//16),
            nn.ReLU(),
        )
        self.dec_deconv = nn.Sequential(
            ResBlock(64),
            ResBlock(64),
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.ReLU(),
                nn.InstanceNorm2d(32)
            ),
            ResBlock(32),
            ResBlock(32),
            nn.Sequential(
                nn.ConvTranspose2d(32, 16, 4, 2, 1),
                nn.ReLU(),
                nn.InstanceNorm2d(16)
            ),
            nn.Sequential(
                nn.Conv2d(16, c_num, 3, padding=1),
                nn.Sigmoid() # cause we want range to be in [0, 1]
            )
        )

    def encode(self, x):
        out = self.enc_conv(x)
        out = out.view(-1, (self.img_h**2)*64//16)
        pre_z = self.enc_linear(out)
        mu = pre_z[:, :self.z_dim]
        log_var = pre_z[:, self.z_dim:]
        return mu, log_var
    
    def decode(self, z):
        out = self.dec_linear(z)
        out = out.view(-1, 64, self.img_h//4, self.img_h//4)
        out = self.dec_deconv(out)
        return out
    
    def reparameterize(self, mu, log_var):
        sigma = torch.exp(log_var/2)
        return mu + sigma * torch.randn(sigma.size()).cuda()
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        rec_x = self.decode(z)
        return rec_x, mu, log_var
                