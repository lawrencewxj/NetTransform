import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import sys
sys.path.append('../')
from net2net_original import wider
import copy
import numpy as np

from utils import NLL_loss_instance
from utils import PlotLearning


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
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
parser.add_argument('--log-interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before logging status')
parser.add_argument('--noise', type=int, default=1,
                    help='noise or no noise 0-1')
parser.add_argument('--weight_norm', type=int, default=1,
                    help='norm or no weight norm 0-1')
parser.add_argument('--plot-name', help='name of the plot (win) to be shown in visdom')
parser.add_argument('--env-name', help='env of the plot in visdom')
parser.add_argument('-v', help='Verbose')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

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


kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=True, download=True, transform=train_transform),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=False, transform=test_transform),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(3, 2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(3, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.AvgPool2d(5, 1)
        self.fc1 = nn.Linear(64 * 3 * 3, 10)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, np.sqrt(2. / n))
                m.bias.data.fill_(0.0)
            if isinstance(m, nn.Linear):
                m.bias.data.fill_(0.0)

    def forward(self, x):
        try:
            x = self.conv1(x)
            x = self.bn1(x)
            x = F.relu(x)
            x = self.pool1(x)
            x = self.conv2(x)
            x = self.bn2(x)
            x = F.relu(x)
            x = self.pool2(x)
            x = self.conv3(x)
            x = self.bn3(x)
            x = F.relu(x)
            x = self.pool3(x)
            x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
            x = self.fc1(x)
            return F.log_softmax(x, dim=1)
        except RuntimeError:
            print(x.size())

    def net2net_wider(self):
        self.conv1, self.conv2, _ = wider(self.conv1, self.conv2, 32,
                                          self.bn1)
        self.conv2, self.conv3, _ = wider(self.conv2, self.conv3, 64,
                                          self.bn2)
        self.conv3, self.fc1, _ = wider(self.conv3, self.fc1, 128,
                                        self.bn3)
        print(self)

    def net2net_deeper(self):
        s = deeper(self.conv1, nn.ReLU, bnorm_flag=True, weight_norm=args.weight_norm, noise=args.noise)
        self.conv1 = s
        s = deeper(self.conv2, nn.ReLU, bnorm_flag=True, weight_norm=args.weight_norm, noise=args.noise)
        self.conv2 = s
        s = deeper(self.conv3, nn.ReLU, bnorm_flag=True, weight_norm=args.weight_norm, noise=args.noise)
        self.conv3 = s
        print(self)

    def net2net_deeper_nononline(self):
        s = deeper(self.conv1, None, bnorm_flag=True, weight_norm=args.weight_norm, noise=args.noise)
        self.conv1 = s
        s = deeper(self.conv2, None, bnorm_flag=True, weight_norm=args.weight_norm, noise=args.noise)
        self.conv2 = s
        s = deeper(self.conv3, None, bnorm_flag=True, weight_norm=args.weight_norm, noise=args.noise)
        self.conv3 = s
        print(self)

    def define_wider(self):
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128*3*3, 10)

    def define_wider_deeper(self):
        self.conv1 = nn.Sequential(nn.Conv2d(3, 12, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(12),
                                   nn.ReLU(),
                                   nn.Conv2d(12, 12, kernel_size=3, padding=1))
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Sequential(nn.Conv2d(12, 24, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(24),
                                   nn.ReLU(),
                                   nn.Conv2d(24, 24, kernel_size=3, padding=1))
        self.bn2 = nn.BatchNorm2d(24)
        self.conv3 = nn.Sequential(nn.Conv2d(24, 48, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(48),
                                   nn.ReLU(),
                                   nn.Conv2d(48, 48, kernel_size=3, padding=1))
        self.bn3 = nn.BatchNorm2d(48)
        self.fc1 = nn.Linear(48*3*3, 10)
        print(self)


def net2net_deeper_recursive(model):
    """
    Apply deeper operator recursively any conv layer.
    """
    for name, module in model._modules.items():
        if isinstance(module, nn.Conv2d):
            s = deeper(module, nn.ReLU, bnorm_flag=False)
            model._modules[name] = s
        elif isinstance(module, nn.Sequential):
            module = net2net_deeper_recursive(module)
            model._modules[name] = module
    return model


def train(epoch):
    model.train()
    avg_loss = 0.0
    avg_accu = 0.0
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
        avg_accu += pred.eq(target.data.view_as(pred)).cpu().sum()
        avg_loss += loss.item()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    avg_loss /= batch_idx + 1
    avg_accu = 100. * avg_accu.item() / len(train_loader.dataset)
    return avg_accu, avg_loss


def test():
    model.eval()
    test_loss = 0.0
    correct = 0.0
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += F.nll_loss(output, target,
                                    reduction='sum').item()  # sum up batch loss
            pred = output.data.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct.item() / len(test_loader.dataset)))
    return 100. * correct.item() / len(test_loader.dataset), test_loss


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
        print 'error'

    return win_accuracy, win_loss



def run_training(model, run_name, epochs, plot=None, win_accuracy=None, win_loss=None):
    visdom_log = {'model_type': run_name, 'epoch': [],
                  'train_accuracy': [], 'test_accuracy': [],
                  'train_loss': [], 'test_loss': [],
                  'top_train_data': {'epoch': 0, 'accuracy': 0.0, 'loss': 0.0},
                  'top_test_data': {'epoch': 0, 'accuracy': 0.0, 'loss': 0.0}}

    global optimizer
    model.cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=0.0001)
    # if plot is None:
    #     plot = PlotLearning('./plots/cifar/', 10, plot_name=run_name)
    for epoch in range(1, epochs + 1):
        accu_train, loss_train = train(epoch)
        accu_test, loss_test = test()
        logs = {}
        logs['acc'] = accu_train
        logs['val_acc'] = accu_test
        logs['loss'] = loss_train
        logs['val_loss'] = loss_test
        # plot.plot(logs)

        visdom_log['epoch'].append(epoch)
        visdom_log['train_accuracy'].append(accu_train)
        visdom_log['test_accuracy'].append(accu_test)
        visdom_log['train_loss'].append(loss_train)
        visdom_log['test_loss'].append(loss_test)

        live_data = {'epoch': epoch, 'train_accuracy': accu_train,
                     'test_accuracy': accu_test, 'train_loss': loss_train,
                     'test_loss': loss_test}

        # Start Plotting live data
        if plot is not None:
            win_accuracy, win_loss = plot_live_data(
                plot, win_accuracy, win_loss, run_name, **live_data)

    return visdom_log, win_accuracy, win_loss


if __name__ == "__main__":
    logs = []
    colors = []
    trace_names = []

    if args.plot_name is not None:
        visdom_live_plot = PlotLearning(
            './plots/cifar/', 10, plot_name=args.plot_name, env_name=args.env_name)
    else:
        visdom_live_plot = None

    start_t = time.time()
    print("\n\n > Teacher training ... ")
    colors.append('orange')
    trace_names.extend(['Teacher Train', 'Teacher Test'])
    model = Net()
    model.cuda()
    criterion = nn.NLLLoss()
    log_base, win_accuracy, win_loss = run_training(model, 'Teacher_', args.epochs, visdom_live_plot)
    logs.append(log_base)

    # wider student training
    print("\n\n > Wider Student training ... ")
    colors.append('blue')
    trace_names.extend(['Wider Net2Net Train', 'Wider Net2Net Test'])
    model_ = Net()
    model_ = copy.deepcopy(model)

    del model
    model = model_
    model.net2net_wider()
    log_net2net, win_accuracy, win_loss = run_training(model, 'Wider_student_', args.epochs, visdom_live_plot, win_accuracy, win_loss)
    logs.append(log_net2net)

    # # wider + deeper student training
    # print("\n\n > Wider+Deeper Student training ... ")
    # model_ = Net()
    # model_.net2net_wider()
    # model_ = copy.deepcopy(model)
    #
    # del model
    # model = model_
    # model.net2net_deeper_nononline()
    # run_training(model, 'WiderDeeper_student_', args.epochs, plot)
    # print(" >> Time tkaen by whole net2net training  {}".format(time.time() - start_t))

    # wider teacher training
    start_t = time.time()
    print("\n\n > Wider teacher training ... ")
    colors.append('green')
    trace_names.extend(['Wider Random Train', 'Wider Random Test'])
    model_ = Net()

    del model
    model = model_
    model.define_wider()
    model.cuda()
    log_random_init, win_accuracy, win_loss = run_training(model, 'Wider_teacher_', args.epochs, visdom_live_plot, win_accuracy, win_loss)
    print(" >> Time taken  {}".format(time.time() - start_t))
    logs.append(log_random_init)

    # # wider deeper teacher training
    # print("\n\n > Wider+Deeper teacher training ... ")
    # start_t = time.time()
    # model_ = Net()
    #
    # del model
    # model = model_
    # model.define_wider_deeper()
    # run_training(model, 'Wider_Deeper_teacher_', args.epochs + 1)
    # print(" >> Time taken  {}".format(time.time() - start_t))

    visdom_plot_final = PlotLearning(
        './plots/cifar/', 10, plot_name='ergol_initial_wider_run1')
    visdom_plot_final.plot_logs(logs, trace_names, colors)
