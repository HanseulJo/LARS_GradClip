import argparse
import time
import logging
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.optim.lr_scheduler as LRscheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import SmallerNet, LargerNet

"""
    references:
    https://github.com/pytorch/examples/blob/master/mnist/main.py

    example run:
    python train.py --batch-size 100 --lr 0.01 --optimizer SGD --dry-run

"""
def argparser(description):
    # Training settings
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--optimizer', type=str, default='SGD', metavar='N',
                        help='optimizer to be used (default: SGD)')
    args = parser.parse_args()
    return args

def train(args, model, device, train_loader, optimizer, epoch,):
    model.train()
    train_loss = 0
    train_acc = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)  # reduction = 'mean'
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(data)
        pred = output.argmax(dim=1, keepdim=True) 
        correct = pred.eq(target.view_as(pred)).sum().item()
        train_acc += correct
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accracy: {}/{} ({:.0f}%)'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(),
                correct, len(data), 100. * correct / len(data)))
            if args.dry_run:
                break
    
    train_loss /= len(train_loader.dataset)
    accuracy = 100. * train_acc / len(train_loader.dataset)
    print('\nTrain set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        train_loss, train_acc, len(train_loader.dataset), accuracy))
    
    return (train_loss, 100. * train_acc / len(train_loader.dataset))
    

def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            test_acc += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, test_acc, len(test_loader.dataset),
        100. * test_acc / len(test_loader.dataset)))

    return (test_loss, 100. * test_acc / len(test_loader.dataset))


def poly_lr_scheduler(optimizer, epoch, max_epoch, lr_init=1, lr_final=1e-4, degree=2):
    """
    Decay learning rate with polynomial of degree 2.
    """
    lr = lr_final + (lr_init - lr_final) * (1 - epoch/max_epoch) ** degree
    print('LR is set to {}'.format(lr))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


if __name__ == '__main__':
    args = argparser('LARS, GradClip with MNIST')
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    #logger = log_maker(test='test')
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset_train = datasets.MNIST('./data', train=True, transform=transform)
    dataset_test = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(dataset_train, **train_kwargs)
    test_loader = DataLoader(dataset_test, **test_kwargs)

    models = {
        'S': SmallerNet(),
        'L': LargerNet(),
    }
    model = models['S'].to(device)

    optimizers = {
        'SGD': optim.SGD(model.parameters(), lr=args.lr),
        'LARS': None,
        'GradClip': None,
    }
    optim_original = optimizers[args.optimizer]

    # learning rate decay: polynomial of degree 2
    losses = {
        'train': [],
        'test': []
    }
    accuracies = {
        'train': [],
        'test': []
    }
    for epoch in range(1, args.epochs + 1):
        optimizer = poly_lr_scheduler(optim_original, epoch-1, args.epochs-1, lr_init=args.lr, lr_final=args.lr*1e-4)
        tr_l, tr_a = train(args, model, device, train_loader, optimizer, epoch)
        losses['train'].append(tr_l)
        accuracies['train'].append(tr_a)
        te_l, te_a = test(model, device, test_loader)
        losses['test'].append(te_l)
        accuracies['test'].append(te_a)

    plt.plot(list(range(1, args.epochs+1)), accuracies['test'])
    plt.show()
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
