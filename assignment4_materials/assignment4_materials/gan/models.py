import torch
from spectral_normalization import SpectralNorm

class Discriminator(torch.nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        
        #Hint: Hint: Apply spectral normalization to convolutional layers. Input to SpectralNorm should be your conv nn module
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        self.conv1 = SpectralNorm(nn.ConvTranspose2d(3, 128, 4, 2, padding=1))
        self.bn1 = nn.BatchNorm2d(3, 128)
        self.conv2 = SpectralNorm(nn.ConvTranspose2d(128, 256, 4, 2, padding=1))
        self.bn2 = nn.BatchNorm2d(128, 256)
        self.conv3 = SpectralNorm(nn.ConvTranspose2d(256, 512, 4, 2, padding=1))
        self.bn3 = nn.BatchNorm2d(256, 512)
        self.conv4 = SpectralNorm(nn.ConvTranspose2d(512, 1024, 4, 2, padding=1))
        self.bn4 = nn.BatchNorm2d(512, 1024)
        self.conv5 = SpectralNorm(nn.ConvTranspose2d(1024, 1, 4, 1, padding=1))
        
        self.out = nn.Tanh()
        

        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
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
        x = self.bn4(x)
        
        
        x = self.conv5(x)
        x = self.out(x)
        
        
        ##########       END      ##########
        
        return x


class Generator(torch.nn.Module):
    def __init__(self, noise_dim, output_channels=3):
        super(Generator, self).__init__()    
        self.noise_dim = noise_dim
    
        ####################################
        #          YOUR CODE HERE          #
        ####################################
        self.conv0 = nn.ConvTranspose2d(noise_dim, 1024, 4, 1)
        self.bn1 = nn.BatchNorm2d(noise_dim, 1024)
        self.conv1 = nn.ConvTranspose2d(1024, 512, 4, 2)
        self.bn1 = nn.BatchNorm2d(1024, 512)
        self.conv2 = nn.ConvTranspose2d(512, 256, 4, 2)
        self.bn2 = nn.BatchNorm2d(512, 256)
        self.conv3 = nn.ConvTranspose2d(256, 128, 4, 2)
        self.bn3 = nn.BatchNorm2d(256, 128)
        self.conv4 = nn.ConvTranspose2d(128, 3, 4, 2)
        self.bn4 = nn.BatchNorm2d(128, 3)
        self.out = nn.Tanh()
        
        ##########       END      ##########
    
    def forward(self, x):
        
        ####################################
        #          YOUR CODE HERE          #
        ####################################
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
        x = self.bn4(x)
        x = self.out(x)
        ##########       END      ##########
        return x
    

