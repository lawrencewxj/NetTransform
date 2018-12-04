import argparse
import time
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import sys
sys.path.append('../')
import copy

from utils import PlotLearning
from convnet import ConvNet, CIFAR10

DISPLAY_INTERVAL = 200
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
parser.add_argument('--plot-name',
                    help='name of the plot (win) to be shown in visdom')
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


kwargs = {'num_workers': 8, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=True, download=True, transform=train_transform),
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.CIFAR10('./data', train=False, transform=test_transform),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


def train(model, epoch):
    # Set the net to train mode. Only applies for certain modules when
    # BatchNorm or Drop outs are used in the net.
    optimizer = optim.SGD(model.parameters(), lr=args.lr,
                          momentum=args.momentum, weight_decay=0.001)

    model.train(mode=True)

    running_loss = 0.0
    num_correct_predictions = 0
    num_train_images = len(train_loader.dataset)

    batch_size = train_loader.batch_size
    for batch_idx, (inputs, targets) in enumerate(train_loader):
        # Transfer the data to GPU if use_cuda is set
        if args.cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        # Reset the gradients to zero for each batch
        inputs, targets = Variable(inputs), Variable(targets)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = model.criterion(outputs, targets)
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
                    epoch, batch_idx * batch_size, num_train_images,
                           100. * batch_idx / len(train_loader),
                    DISPLAY_INTERVAL, loss.item()))

    train_loss = running_loss / num_train_images
    train_accuracy = 100. * num_correct_predictions / num_train_images

    print('\nTraining Set: Avg Loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
        train_loss, num_correct_predictions, num_train_images, train_accuracy))

    return train_accuracy, train_loss


def test(model):
    # Sets the module in evaluation mode. Only applies for certain modules when
    # BatchNorm or Drop outs are used in the net. Undo the effect of
    # net.train(mode=True) while training.
    model.eval()
    running_loss = 0.0
    num_correct_predictions = 0
    num_test_images = len(test_loader.dataset)

    # No gradient calculation required during training
    with torch.no_grad():
        for inputs, targets in test_loader:
            if args.cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = model(inputs)
            _, predicted = outputs.max(1)
            num_correct_predictions += predicted.eq(targets).sum().item()
            loss = model.criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)

    test_loss = running_loss / num_test_images
    test_accuracy = 100. * num_correct_predictions / num_test_images
    print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, num_correct_predictions, num_test_images, test_accuracy))

    return test_accuracy, test_loss


def run_training(model, run_name, epochs, plot=None):
    visdom_log = {'model_type': run_name, 'epoch': [],
                  'train_accuracy': [], 'test_accuracy': [],
                  'train_loss': [], 'test_loss': [],
                  'top_train_data': {'epoch': 0, 'accuracy': 0.0, 'loss': 0.0},
                  'top_test_data': {'epoch': 0, 'accuracy': 0.0, 'loss': 0.0}}

    model.cuda()

    if plot is None:
        plot = PlotLearning('./plots/cifar/', 10, plot_name=run_name)
    for epoch in range(1, epochs + 1):
        accu_train, loss_train = train(model, epoch)
        accu_test, loss_test = test(model)
        logs = {}
        logs['acc'] = accu_train
        logs['val_acc'] = accu_test
        logs['loss'] = loss_train
        logs['val_loss'] = loss_test
        plot.plot(logs)

        visdom_log['epoch'].append(epoch)
        visdom_log['train_accuracy'].append(accu_train)
        visdom_log['test_accuracy'].append(accu_test)
        visdom_log['train_loss'].append(loss_train)
        visdom_log['test_loss'].append(loss_test)

    return plot, visdom_log


if __name__ == "__main__":
    logs = []
    colors = []
    trace_names = []

    start_t = time.time()
    print("\n\n > Teacher training ... ")
    colors.append('orange')
    trace_names.extend(['Teacher Train', 'Teacher Test'])
    model = ConvNet(net_dataset=CIFAR10)
    model.cuda()
    plot, log_base = run_training(model, 'Teacher_', args.epochs)
    logs.append(log_base)

    # wider student training
    print("\n\n > Wider Student training ... ")
    colors.append('blue')
    trace_names.extend(['Wider Net2Net Train', 'Wider Net2Net Test'])
    model_ = ConvNet(net_dataset=CIFAR10)
    model_ = copy.deepcopy(model)

    del model
    model = model_
    model.wider(operation='net2net', widening_factor=2)
    print model
    plot, log_net2net = run_training(model, 'Wider_student_', args.epochs, plot)
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
    model_ = ConvNet(net_dataset=CIFAR10)

    del model
    model = model_
    model.define_wider(widening_factor=2)
    model.cuda()
    print model
    _, log_random_init = run_training(model, 'Wider_teacher_', args.epochs)
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
        './plots/cifar/', 10, plot_name=args.plot_name)
    visdom_plot_final.plot_logs(logs, trace_names, colors)
