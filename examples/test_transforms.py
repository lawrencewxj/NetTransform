from __future__ import division

from torchvision import datasets, transforms
from utils import PlotLearning
import argparse
import copy
import numpy as np
import sys
import torch as th
import torch.nn as nn
import torch.nn.init as init
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
parser.add_argument('--env-name', help='env of the plot in visdom')
parser.add_argument('-v', help='Verbose')

args = parser.parse_args()
use_cuda = not args.no_cuda and th.cuda.is_available()

th.manual_seed(args.seed)
if use_cuda:
    th.cuda.manual_seed(args.seed)

kwargs = {'num_workers': 8, 'pin_memory': True} if use_cuda else {}

train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2023, 0.1994, 0.2010))])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465),
                         std=(0.2023, 0.1994, 0.2010))])

# train_transform = transforms.Compose(
#              [
#               transforms.ToTensor(),
#               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
#
# test_transform = transforms.Compose(
#              [transforms.ToTensor(),
#               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = datasets.CIFAR10(
    DATA_DIRECTORY, train=True, download=True, transform=train_transform)
train_loader = th.utils.data.DataLoader(
    train_set, batch_size=args.batch_size, shuffle=True, **kwargs)

test_set = datasets.CIFAR10(
    DATA_DIRECTORY, train=False, download=True, transform=test_transform)
test_loader = th.utils.data.DataLoader(
    test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        init.xavier_normal_(m.weight.data, gain=init.calculate_gain('relu'))
        m.bias.data.fill_(0.0)


def train(net, optimizer, scheduler, epoch):
    # Set the net to train mode. Only applies for certain modules when
    # BatchNorm or Drop outs are used in the net.
    # scheduler.step()
    net.train(mode=True)

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

        # get max value (currently not used) and corresponding index from
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

    try:
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
    except IOError:
        print 'io error'

    return win_accuracy, win_loss


def start_traning(net, net_type, optimizer, scheduler, plot=None,
                  win_accuracy=None, win_loss=None):
    log = {'model_type': net_type, 'epoch': [],
           'train_accuracy': [], 'test_accuracy': [],
           'train_loss': [], 'test_loss': [],
           'top_train_data': {'epoch': 0, 'accuracy': 0.0, 'loss': 0.0},
           'top_test_data': {'epoch': 0, 'accuracy': 0.0, 'loss': 0.0}}

    for epoch in range(1, args.epochs + 1):
        train_accuracy, train_loss = train(net, optimizer, scheduler, epoch)
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
    colors = []
    trace_names = []

    if args.plot_name is not None:
        visdom_live_plot = PlotLearning(
            './plots/cifar/', 10, plot_name=args.plot_name, env_name=args.env_name)
    else:
        visdom_live_plot = None

    print("\n\n > Teacher (Base Network) training ... ")
    colors.append('orange')
    trace_names.extend(['Teacher Train', 'Teacher Test'])
    teacher_model = ConvNet(net_dataset=CIFAR10)
    # teacher_model.apply(weights_init)
    teacher_model.cuda()
    optimizer = optim.SGD(teacher_model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=0.001) # 0.0001 used in resnet paper for cifar10
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.2)
    scheduler = None

    print teacher_model
    log_base, win_accuracy, win_loss = start_traning(
        teacher_model, 'Teacher', optimizer, scheduler, visdom_live_plot)
    logs.append(log_base)

    # wider student training from Net2Net
    print("\n\n > Wider Student training (Net2Net)... ")
    colors.append('blue')
    trace_names.extend(['Wider Net2Net Train', 'Wider Net2Net Test'])
    n2n_model_wider = copy.deepcopy(teacher_model)
    n2n_model_wider.wider('net2net', widening_factor=2)
    optimizer = optim.SGD(n2n_model_wider.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=0.001)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.2)
    print n2n_model_wider
    log_net2net, win_accuracy, win_loss = start_traning(
        n2n_model_wider, 'WideNet2Net', optimizer, scheduler,
        visdom_live_plot, win_accuracy, win_loss)
    logs.append(log_net2net)

    # wider model training from scratch
    print("\n\n > Wider Network training (Wider Random Init)... ")
    colors.append('green')
    trace_names.extend(['Wider Random Train', 'Wider Random Test'])
    wider_random_init_model = ConvNet(net_dataset=CIFAR10)
    wider_random_init_model.define_wider(widening_factor=2)
    # wider_random_init_model.apply(weights_init)
    wider_random_init_model.cuda()
    optimizer = optim.SGD(wider_random_init_model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=0.001)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.2)
    print wider_random_init_model
    log_random_init, win_accuracy, win_loss = start_traning(
        wider_random_init_model, 'WideRandInit', optimizer, scheduler,
        visdom_live_plot, win_accuracy, win_loss)
    logs.append(log_random_init)

    # wider student training from NetMorph
    # print("\n\n > Wider Student training (NetMorph)... ")
    # colors.append('red')
    # trace_names.extend(['Wider NetMorph Train', 'Wider NetMorph Test'])
    # netmorph_model_wider = ConvNet(net_dataset=CIFAR10)
    # netmorph_model_wider.cuda()
    # netmorph_model_wider = copy.deepcopy(teacher_model)
    # netmorph_model_wider.wider('netmorph', widening_factor=2)
    # optimizer = optim.SGD(netmorph_model_wider.parameters(), lr=args.lr,
    #                       momentum=args.momentum, weight_decay=0.001)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.5)
    # print netmorph_model_wider
    # log_netmorph, win_accuracy, win_loss = start_traning(
    #     netmorph_model_wider, 'WideNetMorph', optimizer, scheduler,
    #     visdom_live_plot, win_accuracy, win_loss)
    # logs.append(log_netmorph)

    # # # deeper model training from scratch
    # print("\n\n > Deeper Network training (Random Init)... ")
    # colors.append('green')
    # trace_names.extend(['Deeper Random Train', 'Deeper Random Test'])
    # deeper_random_init_model = ConvNet(net_dataset=CIFAR10)
    # deeper_random_init_model.define_deeper(deepening_factor=2)
    # deeper_random_init_model.apply(weights_init)
    # deeper_random_init_model.cuda()
    # optimizer = optim.SGD(deeper_random_init_model.parameters(), lr=args.lr,
    #                       momentum=args.momentum, weight_decay=0.001)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
    # print deeper_random_init_model
    # log_random_init, win_accuracy, win_loss = start_traning(
    #     deeper_random_init_model, 'RandInit', optimizer, scheduler,
    #     visdom_live_plot, win_accuracy, win_loss)
    # logs.append(log_random_init)

    # # # # Deeper student training from Net2Net
    # print("\n\n > Deeper Student training (Net2Net)... ")
    # colors.append('blue')
    # trace_names.extend(['Deeper Net2Net Train', 'Deeper Net2Net Test'])
    # n2n_model_deeper = copy.deepcopy(teacher_model)
    # n2n_model_deeper.deeper('net2net')
    # n2n_model_deeper.cuda()
    # optimizer = optim.SGD(n2n_model_deeper.parameters(), lr=args.lr,
    #                       momentum=args.momentum, weight_decay=0.001)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
    # print n2n_model_deeper
    # log_net2net, win_accuracy, win_loss = start_traning(
    #     n2n_model_deeper, 'NetTransform', optimizer, scheduler,
    #     visdom_live_plot, win_accuracy, win_loss)
    # logs.append(log_net2net)

    # Deeper student training from NetMorph
    # print("\n\n > Deeper Student training (NetMorph)... ")
    # colors.append('red')
    # trace_names.extend(['Deeper NetMorph Train', 'Deeper NetMorph Test'])
    # netmorph_model_deeper = copy.deepcopy(teacher_model)
    # netmorph_model_deeper.deeper('net2morph')
    # netmorph_model_deeper.cuda()
    # optimizer = optim.SGD(netmorph_model_deeper.parameters(), lr=args.lr,
    #                       momentum=args.momentum, weight_decay=0.001)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
    # print netmorph_model_deeper
    # log_netmorph, win_accuracy, win_loss = start_traning(
    #     netmorph_model_deeper, 'NetTransform', optimizer, scheduler, visdom_live_plot, win_accuracy, win_loss)
    # logs.append(log_netmorph)

    # wider deeper model training from scratch
    # print("\n\n > Wider Deeper Network training (Random Init)... ")
    # colors.append('green')
    # trace_names.extend(['Wider Deeper Random Train', 'Wider Deeper Random Test'])
    # deeper_wider_random_init_model = ConvNet(net_dataset=CIFAR10)
    # deeper_wider_random_init_model.define_wider(widening_factor=2)
    # deeper_wider_random_init_model.define_deeper(deepening_factor=2)
    # deeper_wider_random_init_model.apply(weights_init)
    # deeper_wider_random_init_model.cuda()
    # optimizer = optim.SGD(deeper_wider_random_init_model.parameters(),
    #                       lr=args.lr,
    #                       momentum=args.momentum, weight_decay=0.001)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
    # print deeper_wider_random_init_model
    # log_random_init, win_accuracy, win_loss = start_traning(
    #     deeper_wider_random_init_model, 'RandInit', optimizer, scheduler,
    #     visdom_live_plot, win_accuracy, win_loss)
    # logs.append(log_random_init)

    # Wider Deeper student training from Net2Net
    # print("\n\n > Wider Deeper Student training (Net2Net)... ")
    # colors.append('blue')
    # trace_names.extend(['Wider Deeper Net2Net Train', 'Wider Deeper Net2Net Test'])
    # n2n_model_deeper_wider = copy.deepcopy(teacher_model)
    # n2n_model_deeper_wider.wider('net2net', widening_factor=2)
    # n2n_model_deeper_wider.deeper('net2net')
    # n2n_model_deeper_wider.cuda()
    # optimizer = optim.SGD(n2n_model_deeper_wider.parameters(), lr=args.lr,
    #                       momentum=args.momentum, weight_decay=0.001)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
    # print n2n_model_deeper_wider
    # log_net2net, win_accuracy, win_loss = start_traning(
    #     n2n_model_deeper_wider, 'WideDeepN2N', optimizer, scheduler, visdom_live_plot, win_accuracy, win_loss)
    # logs.append(log_net2net)

    # # WiderDeeper student training from NetMorph
    # print("\n\n > Wider Deeper Student training (NetMorph)... ")
    # colors.append('red')
    # trace_names.extend(['Wider Deeper NetMorph Train', 'Wider Deeper NetMorph Test'])
    # netmorph_model_deeper_wider = copy.deepcopy(teacher_model)
    # netmorph_model_deeper_wider.deeper('net2morph', widening_factor=2)
    # netmorph_model_deeper_wider.deeper('net2morph')
    # netmorph_model_deeper_wider.cuda()
    # optimizer = optim.SGD(netmorph_model_deeper_wider.parameters(), lr=args.lr,
    #                       momentum=args.momentum, weight_decay=0.001)
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)
    # print netmorph_model_deeper_wider
    # log_netmorph, win_accuracy, win_loss = start_traning(
    #     netmorph_model_deeper_wider, 'NetTransform', optimizer, scheduler, visdom_live_plot, win_accuracy, win_loss)
    # logs.append(log_netmorph)

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
            './plots/cifar/', 10, plot_name=args.plot_name, env_name=args.env_name)
        visdom_plot_final.plot_logs(logs, trace_names, colors)
