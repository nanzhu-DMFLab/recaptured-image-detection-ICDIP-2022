from __future__ import print_function   
import argparse
import os
import myDataset
import networks
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from PIL import Image
from model import MrcNet
from torch.autograd import Variable
from tqdm import tqdm

# GPU ID
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  


# Training settings
parser = argparse.ArgumentParser(description='PyTorch RI vs NI')   
parser.add_argument('--batch-size', type=int, default=64, metavar='N', 
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=32, metavar='N',
                    help='input batch size for testing (default: 32)')
parser.add_argument('--patch-size', type=int, default=96, metavar='N',
                    help='input the patch size of the network during training and testing (default: 96)')
parser.add_argument('--log-dir', default='./log',
                    help='folder to output model checkpoints')
parser.add_argument('--epochs', type=int, default=1200, metavar='N',
                    help='number of epochs to train (default: 1200)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
parser.add_argument('--optimizer', default='sgd', type=str,
                    metavar='OPT', help='The optimizer to use (default: Adagrad)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--loss-adjust', default=100, type=int,
                    help='how many epochs to change the learning rate (default: 400)')
parser.add_argument('--summary-interval', type=int, default=50,
                    help='how many epochs to summary the log')

args = parser.parse_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()


np.random.seed(args.seed)   
if args.cuda:
    cudnn.benchmark = True
kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}


if not os.path.exists(args.log_dir):   
    os.makedirs(args.log_dir) 

# The path of data
data_root = './data/'


# You need to refine this for your data set directory
train_dir = os.path.join(data_root, 'train_image')   # ./data/train_image
val_dir = os.path.join(data_root, 'val_image')

normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

train_loader = myDataset.DataLoaderHalf(
    myDataset.MyDataset(train_dir,
                   transforms.Compose([
                       transforms.RandomCrop(args.patch_size),
                       transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       normalize
                   ])),
    batch_size=args.batch_size, shuffle=False, half_constraint=True,
    sampler_type='RandomBalancedSampler', **kwargs)

val_loader = torch.utils.data.DataLoader(
    myDataset.MyDataset(val_dir,
                    transforms.Compose([
                        transforms.CenterCrop(args.patch_size),
                        transforms.ToTensor(),
                        normalize
                    ])),
    batch_size=args.test_batch_size, shuffle=False, **kwargs)


def main():
    # instantiate model and initialize weights
    model = MrcNet()

    #networks.print_network(model)
    networks.init_weights(model, init_type='normal')

    if args.cuda:
        model.cuda()
    #print('model load!')
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    #L1_criterion = nn.L1Loss(size_average=False).cuda()
    L1_criterion = nn.L1Loss(reduction='mean').cuda()   
    optimizer = create_optimizer(model, args.lr)

    for epoch in range(1, args.epochs+1):
        # update the optimizer learning rate
        adjust_learning_rate(optimizer, epoch)


        train_acc, train_loss = train(train_loader, model, optimizer, criterion, L1_criterion, epoch)
        

        if epoch % args.summary_interval == 0:
            val_acc, val_loss = val(val_loader, model, criterion, epoch)

def train(train_loader, model, optimizer, criterion, L1_criterion, epoch):
    # switch to train mode
    model.train()

    #pbar = tqdm(enumerate(train_loader))   

    running_loss = 0
    running_corrects = 0
 
    #for batch_idx, (data, target) in pbar:   
    for batch_idx, (data, target) in enumerate(train_loader):
                
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data_var, target_var = Variable(data), Variable(target)

        prediction = model(data_var)
        
        _, preds = torch.max(prediction.data, 1)   # return the index of the largest value of a row
        
        loss = criterion(prediction, target_var)

        # statistics
        running_loss += loss.item()   
        #print(running_loss)
        running_corrects += torch.sum(preds == target_var.data).cpu().item()

        # compute gradient and update weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        
        if (batch_idx+1) % 25 == 0:
            
            #pbar.set_description(
            print(
                'Train Epoch: {} [{}/{} ({:.0f}%)]   Loss: {:.6f}'.format(
                    epoch, (batch_idx+1) * len(data_var), len(train_loader.dataset),
                    100. * (batch_idx+1) / len(train_loader),
                    loss.item()))
        

    if epoch % args.log_interval == 0:
        torch.save({'epoch': epoch,
                    'state_dict': model.state_dict()},
                    '{}/checkpoint_{}.pth'.format(args.log_dir, epoch))

    running_loss = running_loss / (len(train_loader.dataset) // args.batch_size)
    ave_corrects = 100. * running_corrects / len(train_loader.dataset)
    print('Train Epoch {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        epoch, running_loss, running_corrects, len(train_loader.dataset), ave_corrects))
    return ave_corrects, running_loss


def val(val_loader, model, criterion, epoch):
    # switch to evaluate mode
    model.eval()

    test_loss = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(val_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        
        # compute output
        output = model(data)
        test_loss += criterion(output, target).item()    # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1]    # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum().item()

    test_loss = test_loss / (len(val_loader.dataset) // args.test_batch_size)
    ave_correct = 100. * correct / len(val_loader.dataset)
    print('Test Epoch: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        test_loss, correct, len(val_loader.dataset), ave_correct))
    return ave_correct, test_loss


def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 100 epochs"""
    lr = args.lr * (0.1 ** (epoch // args.loss_adjust))   
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def create_optimizer(model, new_lr):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=new_lr,
                              momentum=0.9, weight_decay=args.wd)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=new_lr,
                               weight_decay=args.wd, betas=(args.beta1, 0.999))
    elif args.optimizer == 'adagrad':
        optimizer = optim.Adagrad(model.parameters(), lr=new_lr,
                                  lr_decay=args.lr_decay, weight_decay=args.wd)
    return optimizer


if __name__ == '__main__':
    main()
