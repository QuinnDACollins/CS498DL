import torch
from torch.nn.functional import binary_cross_entropy_with_logits as bce_loss
device = torch.device("cuda:0")
def discriminator_loss(logits_real, logits_fake):
    """
    Computes the discriminator loss.
    
    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits 
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.
    
    Inputs:
    - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
    """
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    
    D_x = bce_loss(logits_real, torch.ones(logits_real.size()).to(device))
    D_G_x = bce_loss(logits_fake, torch.zeros(logits_fake.size()).to(device))
    total = D_x + D_G_x
    total.mean()
    ##########       END      ##########
    return total

def generator_loss(logits_fake):
    """
    Computes the generator loss.
    
    You should use the stable torch.nn.functional.binary_cross_entropy_with_logits 
    loss rather than using a separate softmax function followed by the binary cross
    entropy loss.

    Inputs:
    - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Returns:
    - loss: PyTorch Tensor containing the (scalar) loss for the generator.
    """
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    D_G_x = bce_loss(logits_fake, torch.ones(logits_fake.size()).to(device))
    total = D_G_x.mean()
    ##########       END      ##########
    
    return total


def ls_discriminator_loss(scores_real, scores_fake):
    """
    Compute the Least-Squares GAN loss for the discriminator.
    
    Inputs:
    - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    
    loss = None
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    
    D_x = bce_loss(scores_real, torch.ones(scores_real.size()).to(device))
    D_G_x = bce_loss(scores_fake, torch.zeros(scores_fake.size()).to(device))
    total = D_x + D_G_x
    total = 0.5 * total
    total.mean()
    ##########       END      ##########
    return total

def ls_generator_loss(scores_fake):
    """
    Computes the Least-Squares GAN loss for the generator.
    
    Inputs:
    - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
    
    Outputs:
    - loss: A PyTorch Tensor containing the loss.
    """
    
    loss = None
    
    ####################################
    #          YOUR CODE HERE          #
    ####################################
    D_G_x = bce_loss(scores_fake, torch.ones(scores_fake.size()).to(device))
    total = 0.5 * D_G_x.mean()
    ##########       END      ##########
    
    return total
