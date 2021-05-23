import argparse
import logging
import os
import time
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models import SmallerNet, LargerNet
from optimizers import *

"""
    references:
    https://github.com/pytorch/examples/blob/master/mnist/main.py

    example run:
    python train.py --batch-size 100 --lr 0.01 --optimizer SGD --dry-run

"""
def argparser(description):
    # Training settings
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for training (default: 100)')
    parser.add_argument('--test-batch-size', type=int, default=10000, metavar='N',
                        help='input batch size for testing (default: 10000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
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
    parser.add_argument('--optimizer', type=str, default='SGD', metavar='OPTIM',
                        help='optimizer to be used (default: SGD)')
    parser.add_argument('--use-largernet', action='store_true', default=False,
                        help='use LargerNet instead of SmallerNet')
    parser.add_argument('--log-file-on', action='store_true', default=False,
                        help='for print_in_file option to be True')
    parser.add_argument('--eta', type=float, default=0.01, metavar='ETA',
                        help='LARS coefficient(default: 0.01)')
    parser.add_argument('--clip', type=float, default=1., metavar='CLIP',
                        help='gradient clip threshold (default: 1.0')
    parser.add_argument('--lr-decay-degree', type=float, default=2, metavar='CLIP',
                        help='LR scheduling function degree (default: 2')
    args = parser.parse_args()
    return args


def log_maker(args, print_in_file=True, print_hyperparam=True):
    LOG_FORMAT = "[%(asctime)-10s] %(name)s:%(levelname)s - %(message)s"
    logging.basicConfig(format="%(message)s")   # New logs are also printed on the console
    logger = logging.getLogger(args.optimizer if args is not None else "TUNING")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter(fmt=LOG_FORMAT, style="%")

    if print_in_file:
        file_handler = logging.FileHandler(filename=os.path.join('.','logs','info.log'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if print_hyperparam:
        logger.info("Hyperparameters:")
        arglist = [(p,v) for p, v in vars(args).items()]
        arglist.sort(key=lambda x: len(x[0]))
        for p, v in arglist:
            logger.info(f"{p} : \t{v}")
    return logger


def train(args, model, device, train_loader, optimizer, epoch, logger=None):
    model.train()
    train_loss = 0
    train_acc = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)  # reduction = 'mean'
        if loss.isnan():
            raise ValueError("NAN Loss")
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(data)
        pred = output.argmax(dim=1, keepdim=True) 
        correct = pred.eq(target.view_as(pred)).sum().item()
        train_acc += correct
        if batch_idx % args.log_interval == 0:
            if logger is not None:
                logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accracy: {}/{} ({:.0f}%)'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item(),
                    correct, len(data), 100. * correct / len(data)))
            if args.dry_run:
                break
    
    train_loss /= len(train_loader.dataset)
    accuracy = 100. * train_acc / len(train_loader.dataset)
    if logger is not None:
        logger.info('')
        logger.info('Train set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
            train_loss, train_acc, len(train_loader.dataset), accuracy))
    
    return (train_loss, 100. * train_acc / len(train_loader.dataset))
    

def test(args, model, device, test_loader, logger=None):
    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target, reduction='sum')  # sum up batch loss
            if loss.isnan():
                raise ValueError("NAN Loss")
            test_loss += loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            test_acc += pred.eq(target.view_as(pred)).sum().item()
            if args.dry_run:
                break

    test_loss /= len(test_loader.dataset)

    if logger is not None:
        logger.info('')
        logger.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.4f}%)'.format(
            test_loss, test_acc, len(test_loader.dataset),
            100. * test_acc / len(test_loader.dataset)))
        logger.info('')

    return (test_loss, 100. * test_acc / len(test_loader.dataset))


def poly_lr_scheduler(optimizer, epoch, max_epoch, lr_init=1, lr_final=None, degree=2, logger=None):
    """
    Decay learning rate with polynomial of degree 2.
    """
    if lr_final is None:
        lr_final = lr_init * 1e-4
    lr = lr_final + (lr_init - lr_final) * (1 - epoch/max_epoch) ** degree
    if logger is not None:
        logger.info(f'LR is set to {lr}')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def poly_decay(epoch, max_epoch, lr_init, lr_final=None, degree=2):
    if degree <= 0:
        assert degree == 0
        return 1
    if lr_final is None:
        lr_final = lr_init * 1e-4
    lr = lr_final + (lr_init - lr_final) * (1 - epoch/max_epoch) ** degree
    return lr / lr_init


def main():
    args = argparser('LARS, GradClip with MNIST')
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    logger = log_maker(args, print_in_file=args.log_file_on)
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
    dataset_train = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset_test = datasets.MNIST('./data', train=False, transform=transform)
    train_loader = DataLoader(dataset_train, **train_kwargs)
    test_loader = DataLoader(dataset_test, **test_kwargs)

    model = None
    if args.use_largernet:
        model = LargerNet().to(device)
    else:
        model = SmallerNet().to(device)

    optimizer = None
    if args.optimizer == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=args.lr)
    elif args.optimizer == 'LARS':
        optimizer = LARS(model.parameters(), lr=args.lr, eta=args.eta)
    elif args.optimizer == 'GradClip': 
        optimizer = GradClip(model.parameters(), lr=args.lr, threshold=args.clip)

    scheduler = lr_scheduler.LambdaLR(optimizer,
                                      lr_lambda=lambda x: poly_decay(x, args.epochs, args.lr, 
                                                                     degree=args.lr_decay_degree))

    # learning rate decay: polynomial of degree 2
    losses = {'train': [],'test': []}
    accuracies = {'train': [],'test': []}
    domain_length = 0
    interrupt_flag = 0
    try:
        t_start = time.time()
        for epoch in range(1, args.epochs + 1):
            try:
                logger.info(f"LR is modified to {scheduler.get_last_lr()[0]:.4f}")
                tr_l, tr_a = train(args, model, device, train_loader, optimizer, epoch, logger=logger)
                losses['train'].append(tr_l)
                accuracies['train'].append(tr_a)
                te_l, te_a = test(args, model, device, test_loader, logger=logger)
                losses['test'].append(te_l)
                accuracies['test'].append(te_a)
                domain_length += 1
                scheduler.step()
            except KeyboardInterrupt:
                interrupt_flag = 1
                break
        t_train = time.time() - t_start
        logger.info(f"Training Time Lapse: {t_train:.4f} seconds\n")
        time_str = time.strftime("%y-%m-%d %X", time.localtime(t_start))  # format has changed: original: %c
    except Exception as e:
        logger.info("!!! Exception occured !!!")
        logger.info(e)
        logger.info("\n")
    
    if interrupt_flag == 1:
        logger.info("\n!!! Interrupted by Keyboard !!!\n")
        raise InterruptedError

    if domain_length > 0:
        plt.figure("Train Loss")
        plt.plot(list(range(1, domain_length+1)), losses['train'])
        plt.title(f"Train Loss: {args.optimizer} (batchsize {args.batch_size}) (lr_init {args.lr})")
        plt.savefig(os.path.join(".","plots",time_str+' trainloss.png'), dpi=300)
        plt.figure("Test Loss")
        plt.plot(list(range(1, domain_length+1)), losses['test'])
        plt.title(f"Test Loss: {args.optimizer} (batchsize {args.batch_size}) (lr_init {args.lr})")
        plt.savefig(os.path.join(".","plots",time_str+' test_loss.png'), dpi=300)
        plt.figure("Train Accuracy")
        plt.plot(list(range(1, domain_length+1)), accuracies['train'])
        plt.title(f"Train Acc: {args.optimizer} (batchsize {args.batch_size}) (lr_init {args.lr})")
        plt.savefig(os.path.join(".","plots",time_str+' trainaccu.png'), dpi=300)
        plt.figure("Test Accuracy")
        plt.plot(list(range(1, domain_length+1)), accuracies['test'])
        plt.title(f"Test Acc: {args.optimizer} (batchsize {args.batch_size}) (lr_init {args.lr})")
        plt.savefig(os.path.join(".","plots",time_str+' test_accu.png'), dpi=300)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_sgd.pt")

if __name__ == '__main__':
    main()
