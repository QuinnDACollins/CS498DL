import torch
import sys
import torch.nn.functional as F
sys.path.insert(0,"/home/quinndacollins/CS498DL/assignment4_materials/assignment4_materials/gan")
from spectral_normalization import SpectralNorm

class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        
        #Hint: Hint: Apply spectral normalization to convolutional layers. Input to SpectralNorm should be your conv nn module
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        self.slope = 0.2

        self.net = torch.nn.Sequential(
          SpectralNorm(torch.nn.Conv2d(input_channels, 128, 4, 2, padding=1)),
          torch.nn.BatchNorm2d(128),
          torch.nn.LeakyReLU(self.slope),
          SpectralNorm(torch.nn.Conv2d(128, 256, 4, 2, padding=1)),
          torch.nn.BatchNorm2d(256),
          torch.nn.LeakyReLU(self.slope),
          SpectralNorm(torch.nn.Conv2d(256, 512, 4, 2, padding=1)),
          torch.nn.BatchNorm2d(512),
          torch.nn.LeakyReLU(self.slope),
          SpectralNorm(torch.nn.Conv2d(512, 1024, 4, 2, padding=1)),
          torch.nn.BatchNorm2d(1024),
          torch.nn.LeakyReLU(self.slope),
          SpectralNorm(torch.nn.Conv2d(1024, 1, 4, 1, padding=1)),
          torch.nn.LeakyReLU(self.slope)
        )
        
        
        

        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        #convlayer1
        self.net(x)
        
        
        ##########       END      ##########
        
        return x


class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()    
        self.noise_dim = noise_dim
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        self.conv0 = torch.nn.ConvTranspose2d(self.noise_dim, 1024, 4, 1, padding = 1)
        self.bn0 = torch.nn.BatchNorm2d(1024)
        self.conv1 = torch.nn.ConvTranspose2d(1024, 512, 4, 2)
        self.bn1 = torch.nn.BatchNorm2d(512)
        self.conv2 = torch.nn.ConvTranspose2d(512, 256, 4, 2)
        self.bn2 = torch.nn.BatchNorm2d(256)
        self.conv3 = torch.nn.ConvTranspose2d(256, 128, 4, 2)
        self.bn3 = torch.nn.BatchNorm2d(128)
        self.conv4 = torch.nn.ConvTranspose2d(128, 3, 4, 2)
        self.out = torch.nn.Tanh()
        
        ##########       END      ##########
    
    def forward(self, x):
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        orig_shape = x.shape
        x = x.view(-1, self.noise_dim, 1, 1)
        x = self.conv0(x)
        x = self.bn0(x)
        x = F.relu(x)
        #convlayer1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        #convlayer2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        #convlayer3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        #convlayer4 -> Tanh Last Activation
        x = self.conv4(x)
        x = self.out(x)

        ##########       END      ##########
        return x
    

