#import libraries
import time
import torch.backends.cudnn as cudnn
import torch
from torch import nn
from models import SRResNet
from datasets import SRDataset
from utils import *

# Data parameters
base_path = './' 
data_folder = base_path+'Dataset/train' # base path of files
crop_size = 96  # crop size of target HR images
scaling_factor = 4  # Upscaling factor for which model being created

# Model parameters
large_kernel_size = 9  # First Convolution layer kernel size
small_kernel_size = 3  # Kernel size of all other Convolution layers
n_channels = 64  # Channels in-between, the input and output channels
n_blocks = 16  # Count of residual blocks

# Learning parameters
checkpoint = None  # path initialized if needed for retraining 
batch_size = 16  # batch size
start_epoch = 0  
epochs = 50 # Number of epoch in model training
workers = 4  # Workers for loading data in DataLoader
print_freq = 500  
lr = 1e-4  # learning rate
grad_clip = None  # clip if gradients are exploding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cudnn.benchmark = True


def main():

    global start_epoch, epoch, checkpoint

    # Initialize parameters
    if checkpoint is None:
        model = SRResNet(large_kernel_size=large_kernel_size, small_kernel_size=small_kernel_size,n_channels=n_channels, n_blocks=n_blocks, scaling_factor=scaling_factor)
        
        # Optimizers
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),lr=lr)

    # Checkpoint loading parameters
    else:
      
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    # Move to default device
    model = model.to(device)
    criterion = nn.MSELoss().to(device)

    # Train dataloaders
    train_dataset = SRDataset(data_folder,split='train',crop_size=crop_size,scaling_factor=scaling_factor,
                              lr_img_type='imagenet-norm',hr_img_type='[-1, 1]')
    
    # Test dataloaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,pin_memory=True)

    # Start model training based on epochs
    for epoch in range(start_epoch, epochs):
        # One epoch's training
        train(train_loader=train_loader,model=model,criterion=criterion,optimizer=optimizer,epoch=epoch)

        # Save checkpoint
        torch.save({'epoch': epoch,'model': model,'optimizer': optimizer},'checkpoint_srresnet.pth.tar')


def train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.
    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: content loss function (Mean Squared-Error loss)
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  

    batch_time = AverageMeter()
    data_time = AverageMeter() 
    losses = AverageMeter()  

    start = time.time()

    for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        lr_imgs = lr_imgs.to(device)  
        hr_imgs = hr_imgs.to(device)  

        # Forward propogation
        sr_imgs = model(lr_imgs)  

        # Loss out
        loss = criterion(sr_imgs, hr_imgs)

        # Backward propogation
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients 
        if grad_clip is not None:
            clip_gradient(optimizer, grad_clip)

        # Update model
        optimizer.step()

        # Keep track of loss
        losses.update(loss.item(), lr_imgs.size(0))

        # Keep track of batch time
        batch_time.update(time.time() - start)

        # Reset start time
        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]----'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})----'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})----'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader),batch_time=batch_time,data_time=data_time, loss=losses))

    del lr_imgs, hr_imgs, sr_imgs  # cache clearing to make space


if __name__ == '__main__':
    main()
