from __future__ import division

from torchvision import datasets, transforms, utils
from utils import PlotLearning, progress_bar
import argparse
import copy
import numpy as np
import sys
import torch as th
import torch.optim as optim

sys.path.append('../')
from convnet import ConvNet, CIFAR10

DATA_DIRECTORY = './data'
DISPLAY_INTERVAL = 200

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging status')
parser.add_argument('--noise', type=int, default=1,
                    help='noise or no noise 0-1')
parser.add_argument('--weight-norm', type=int, default=1,
                    help='norm or no weight norm 0-1')
parser.add_argument('--plot-name', help='name of the plot (win) to be shown in visdom')
parser.add_argument('-v', help='Verbose')

args = parser.parse_args()
use_cuda = not args.no_cuda and th.cuda.is_available()

th.manual_seed(args.seed)
if use_cuda:
    th.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])  # normalize the data to [-1,1]

train_set = datasets.CIFAR10(
    DATA_DIRECTORY, train=True, download=True, transform=transform)
train_loader = th.utils.data.DataLoader(
    train_set, batch_size=args.batch_size, shuffle=True, **kwargs)

test_set = datasets.CIFAR10(
    DATA_DIRECTORY, train=False, download=True, transform=transform)
test_loader = th.utils.data.DataLoader(
    test_set, batch_size=args.batch_size, shuffle=False, **kwargs)


def train(net, epoch):
    # Set the net to train mode. Only applies for certain modules when
    # BatchNorm or Drop outs are used in the net.
    net.train(mode=True)
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=0.001)

    running_loss = 0.0
    num_correct_predictions = 0
    num_train_images = len(train_loader.dataset)

    batch_size = train_loader.batch_size
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Transfer the data to GPU if use_cuda is set
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        # Reset the gradients to zero for each batch
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = net.criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)

        # get max value (not required) and corresponding index from
        # the calculated outputs (per batch)
        _, predicted = outputs.max(1)
        num_correct_predictions += predicted.eq(targets).sum().item()
        # progress_bar(
        #     batch_idx, len(train_loader),
        #     'Loss: %.3f | Acc: %.3f%% (%d/%d)' % (train_loss / (batch_idx + 1), 100. * correct / total, correct, total))

        if batch_idx % DISPLAY_INTERVAL == DISPLAY_INTERVAL - 1:
            print(
                'Epoch: {} [{}/{} ({:.0f}%)]\tLoss (Batch Size:{}): {:.3f}'.format(
                epoch, batch_idx * batch_size, num_train_images, 100. * batch_idx / len(train_loader),
                DISPLAY_INTERVAL, loss.item()))

    train_loss = running_loss / num_train_images
    train_accuracy = 100. * num_correct_predictions / num_train_images

    print('\nTraining Set: Avg Loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        train_loss, num_correct_predictions, num_train_images, train_accuracy))

    return train_accuracy, train_loss


def test(net, verbose=False):
    # Sets the module in evaluation mode. Only applies for certain modules when
    # BatchNorm or Drop outs are used in the net. Undo the effect of
    # net.train(mode=True) while training.
    net.eval()
    running_loss = 0.0
    num_correct_predictions = 0
    num_test_images = len(test_loader.dataset)

    # No gradient calculation required during training
    with th.no_grad():
        for inputs, targets in test_loader:
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = net(inputs)
            _, predicted = outputs.max(1)
            num_correct_predictions += predicted.eq(targets).sum().item()
            loss = net.criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)

    test_loss = running_loss / num_test_images
    test_accuracy = 100. * num_correct_predictions / num_test_images
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, num_correct_predictions, num_test_images, test_accuracy))

    if verbose:
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        with th.no_grad():
            for inputs, targets in test_loader:
                if use_cuda:
                    inputs, targets = inputs.cuda(), targets.cuda()

                outputs = net(inputs)
                _, predicted = th.max(outputs, 1)
                c = (predicted == targets).squeeze()
                for i in range(4):
                    label = targets[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1

        for i in range(10):
            print('Accuracy of %5s : %2d %%' % (
                CIFAR10.CLASSES[i], 100 * class_correct[i] / class_total[i]))

    return test_accuracy, test_loss


def plot_live_data(plot, win_accuracy, win_loss, net_type, epoch,
                   train_accuracy, test_accuracy, train_loss, test_loss):

    if win_accuracy is None:
        win_accuracy = plot.viz.line(
            X=np.array([epoch]), Y=np.array([train_accuracy]),
            name=net_type + '_TA',
            opts=dict(xlabel='Epochs', ylabel='Accuracy',
                      title='Accuracy Vs Epoch - ' + plot.plot_name)
        )
        plot.viz.line(X=np.array([0]), Y=np.array([test_accuracy]),
                      win=win_accuracy,
                      name=net_type + '_VA', update='append')
    else:
        plot.viz.line(X=np.array([epoch]), Y=np.array([train_accuracy]),
                      name=net_type + '_TA', win=win_accuracy,
                      update='append')
        plot.viz.line(X=np.array([epoch]), Y=np.array([test_accuracy]),
                      name=net_type + '_VA', win=win_accuracy,
                      update='append')

    if win_loss is None:
        win_loss = plot.viz.line(
            X=np.array([epoch]), Y=np.array([train_loss]),
            name=net_type + '_TL',
            opts=dict(xlabel='Epochs', ylabel='Loss',
                      title='Loss Vs Epoch - ' + plot.plot_name)
        )
        plot.viz.line(X=np.array([epoch]), Y=np.array([test_loss]),
                      win=win_loss,
                      name=net_type + '_VL', update='append')
    else:
        plot.viz.line(X=np.array([epoch]), Y=np.array([train_loss]),
                      name=net_type + '_TL', win=win_loss,
                      update='append')
        plot.viz.line(X=np.array([epoch]), Y=np.array([test_loss]),
                      name=net_type + '_VL', win=win_loss,
                      update='append')

    return win_accuracy, win_loss


def start_traning(net, net_type, plot=None, win_accuracy=None, win_loss=None):
    log = {'model_type': net_type, 'epoch': [],
           'train_accuracy': [], 'test_accuracy': [],
           'train_loss': [], 'test_loss': [],
           'top_train_data': {'epoch': 0, 'accuracy': 0.0, 'loss': 0.0},
           'top_test_data': {'epoch': 0, 'accuracy': 0.0, 'loss': 0.0}}

    for epoch in range(1, args.epochs + 1):
        train_accuracy, train_loss = train(net, epoch)
        test_accuracy, test_loss = test(net)

        log['epoch'].append(epoch)
        log['train_accuracy'].append(train_accuracy)
        log['test_accuracy'].append(test_accuracy)
        log['train_loss'].append(train_loss)
        log['test_loss'].append(test_loss)

        if test_accuracy > log['top_test_data']['accuracy']:
            log['top_test_data']['epoch'] = epoch
            log['top_test_data']['accuracy'] = test_accuracy
            log['top_test_data']['loss'] = test_loss

        if train_accuracy > log['top_train_data']['accuracy']:
            log['top_train_data']['epoch'] = epoch
            log['top_train_data']['accuracy'] = train_accuracy
            log['top_train_data']['loss'] = train_loss

        live_data ={'epoch': epoch, 'train_accuracy': train_accuracy,
                    'test_accuracy': test_accuracy, 'train_loss': train_loss,
                    'test_loss': test_loss}

        # Start Plotting live data
        if plot is not None:
            win_accuracy, win_loss = plot_live_data(
                plot, win_accuracy, win_loss, net_type, **live_data)

    return log, win_accuracy, win_loss


if __name__ == "__main__":

    logs = []

    if args.plot_name is not None:
        visdom_live_plot = PlotLearning('./plots/cifar/', 10, prefix='NetMorph ',
                                        plot_name='NetMorph(' + args.plot_name + ')')
    else:
        visdom_live_plot = None

    print("\n\n > Teacher (Base Network) training ... ")
    teacher_model = ConvNet(CIFAR10)
    teacher_model.cuda()
    print teacher_model
    print(type(teacher_model))
    log_base, win_accuracy, win_loss = start_traning(
        teacher_model, 'Teacher', visdom_live_plot)
    logs.append(log_base)

    teacher_model.net2net_deeper()
    exit()
    # wider model training from scratch
    print("\n\n > Wider Network training (Wider Random Init)... ")
    wider_random_init_model = ConvNet(CIFAR10)
    wider_random_init_model.define_wider(widening_factor=2)
    wider_random_init_model.cuda()
    print wider_random_init_model
    log_random_init, win_accuracy, win_loss = start_traning(
        wider_random_init_model, 'WideRandInit',
        visdom_live_plot, win_accuracy, win_loss)
    logs.append(log_random_init)

    # wider student training from NetMorph
    print("\n\n > Wider Student training (NetMorph)... ")
    model_wider = ConvNet(CIFAR10)
    model_wider.cuda()
    model_wider = copy.deepcopy(teacher_model)
    model_wider.netmorph_wider(widening_factor=2)
    print model_wider
    log_netmorph, win_accuracy, win_loss = start_traning(
        model_wider, 'WideNetMorph', visdom_live_plot, win_accuracy, win_loss)
    logs.append(log_netmorph)

    for log in logs:
        print '*' * 30
        print log['model_type']
        print '*' * 30
        print 'Best Training Stats...'
        print 'epoch: {}'.format(log['top_train_data']['epoch'])
        print 'accuracy:' + str(log['top_train_data']['accuracy'])
        print 'loss:' + str(log['top_train_data']['loss'])
        print '-' * 30
        print 'Best Test Stats...'
        print 'epoch:' + str(log['top_test_data']['epoch'])
        print 'accuracy:' + str(log['top_test_data']['accuracy'])
        print 'loss:' + str(log['top_test_data']['loss'])
        print '\n'

    if args.plot_name is not None:
        visdom_plot_final = PlotLearning(
            './plots/cifar/', 10, prefix='NetMorph_wider',
            plot_name=args.plot_name + "_wider")
        visdom_plot_final.plot_logs(logs, args.plot_name)



