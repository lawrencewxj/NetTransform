from __future__ import division

from torchvision import datasets, transforms
from utils import PlotLearning
import argparse
import copy
import numpy as np
import sys
import torch as th
import time
import os
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim

sys.path.append('../')
from convnet import ConvNet, CIFAR10
import im2col

DATA_DIRECTORY = './data'
MODEL_PATH = './best_model'
DISPLAY_INTERVAL = 200

# Training settings
parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=256, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging status')
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

train_set = datasets.CIFAR10(
    DATA_DIRECTORY, train=True, download=True, transform=train_transform)
train_loader = th.utils.data.DataLoader(
    train_set, batch_size=args.batch_size, shuffle=True, **kwargs)

test_set = datasets.CIFAR10(
    DATA_DIRECTORY, train=False, download=True, transform=test_transform)
test_loader = th.utils.data.DataLoader(
    test_set, batch_size=args.test_batch_size, shuffle=False, **kwargs)


# def weights_init(m):
#     if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
#         # init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
#         init.xavier_normal_(m.weight.data, gain=init.calculate_gain('relu'))
#         m.bias.data.fill_(0.0)


def train(net, optimizer, scheduler, epoch):
    # Set the net to train mode. Only applies for certain modules when
    # BatchNorm or Drop outs are used in the net.
    scheduler.step()
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
                batch_size, loss.item()))

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


def start_training(net, net_type, optimizer, scheduler, plot=None,
                   win_accuracy=None, win_loss=None):
    log = {'model_type': net_type, 'epoch': [],
           'train_accuracy': [], 'test_accuracy': [],
           'train_loss': [], 'test_loss': [],
           'top_train_data': {'epoch': 0, 'accuracy': 0.0, 'loss': 0.0},
           'top_test_data': {'epoch': 0, 'accuracy': 0.0, 'loss': 0.0}}

    for epoch in range(1, args.epochs + 1):
        train_accuracy, train_loss = train(net, optimizer, scheduler, epoch)
        test_accuracy, test_loss = test(net)
        # scheduler.step(test_accuracy)

        # print optimizer.state_dict()
        # print scheduler.state_dict()

        log['epoch'].append(epoch)
        log['train_accuracy'].append(train_accuracy)
        log['test_accuracy'].append(test_accuracy)
        log['train_loss'].append(train_loss)
        log['test_loss'].append(test_loss)

        if test_accuracy > log['top_test_data']['accuracy']:
            log['top_test_data']['epoch'] = epoch
            log['top_test_data']['accuracy'] = test_accuracy
            log['top_test_data']['loss'] = test_loss

        # if train_accuracy > log['top_train_data']['accuracy']:
            log['top_train_data']['epoch'] = epoch
            log['top_train_data']['accuracy'] = train_accuracy
            log['top_train_data']['loss'] = train_loss

            if net_type == 'Teacher':
                th.save(
                    net.state_dict(),
                    os.path.join(MODEL_PATH,
                                 '_' if args.plot_name is None else args.plot_name + '_bestmodel.pt'))

        live_data = {'epoch': epoch, 'train_accuracy': train_accuracy,
                     'test_accuracy': test_accuracy, 'train_loss': train_loss,
                     'test_loss': test_loss}

        # Start Plotting live data
        if plot is not None:
            win_accuracy, win_loss = plot_live_data(
                plot, win_accuracy, win_loss, net_type, **live_data)

    return log, win_accuracy, win_loss


def get_optimizer(model):
    # wt decay = 0.0001 in resnet paper, 0.0005 used in wide-resnet for cifar10
    return optim.SGD(model.parameters(), lr=args.lr,
                     momentum=args.momentum, weight_decay=0.0005)


def get_scheduler(optimizer):
    return optim.lr_scheduler.StepLR(optimizer, step_size=45, gamma=0.2)
    return optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, mode='max', verbose=True, patience=15)


def save_optimizer_scheduler(optimizer, scheduler, net_type):
    optim_scheduler = {'optimizer': optimizer.state_dict(),
                       'scheduler': scheduler.state_dict()}
    th.save(optim_scheduler,
            os.path.join(
                MODEL_PATH,
                net_type + '_' + '' if args.plot_name is None else args.plot_name + '_optim_scheduler.pt'))


if __name__ == "__main__":

    np.set_printoptions(threshold=np.inf)

    # example 1: single channel
    # img = np.array([[[[6, 2, 2], [5, 8, 7], [1, 4, 3]]]])
    #
    # kernel = np.array(
    #     [[[[0, -1], [1, 0]], [[5, 4], [3, 2]], [[16, 24], [68, -2]]],
    #      [[[60, 22], [32, 18]], [[35, 46], [7, 23]], [[78, 81], [20, 42]]]])

    # img = np.random.rand(4, 4, 3, 3)
    # # print img
    # print img.shape
    # img_col = im2col.im2col(img, 3, 3, padding=2)
    # # print img_col
    # print img_col.shape
    # img2 = im2col.col2im(img_col, (4, 4, 3, 3), 3, 3, padding=2)
    # img2 = img2/9
    # img_col2 = im2col.im2col(img, 3, 3, padding=2)
    # print img_col[1]
    # print img_col2[1]
    # # print img == img2
    # exit()
    #
    # # img_col = im2col.im2col_indices(img, 2, 2, padding=0) # removing padding for im2col and col2im
    # kernel_col = kernel.reshape(2, -1)  # 2 is number of filters
    #
    # prod = np.matmul(img_col, kernel_col.T) # for im2col and col2im methods
    # # prod = np.matmul(kernel_col, img_col)
    # col_to_prod = im2col.col2im(prod.T, (2, 1, 2, 2), 2, 2)
    # # col_to_prod = im2col.col2im_indices(prod.T, (2, 1, 2, 2), 2, 2, padding=0)
    # print col_to_prod
    # exit()

    # # example 2: multi channel
    # img = np.array([[[[16, 24, 32], [47, 18, 26], [68, 12, 9]],
    #                  [[26, 57, 43], [24, 21, 12], [2, 11, 19]],
    #                  [[18, 47, 21], [4, 6, 12], [81, 22, 13]]]])
    #
    # kernel = np.array(
    #     [[[[0, -1], [1, 0]], [[5, 4], [3, 2]], [[16, 24], [68, -2]]],
    #      [[[60, 22], [32, 18]], [[35, 46], [7, 23]], [[78, 81], [20, 42]]]])
    #
    # print '*' * 30
    # print img
    # print '*' * 30
    # img_col = im2col.im2col(img, 2, 2)
    # print '*' * 30
    # print img_col
    # print '*' * 30
    #
    #
    # img = im2col.col2im(img_col, (1, 3, 3, 3), 2, 2)
    # print '*' * 30
    # print img/[[1, 2, 1], [2, 4, 2], [1, 2, 1]]
    # print '*' * 30
    # exit()

    # # img_col = im2col.im2col_indices(img, 2, 2, padding=0) # removing padding for im2col and col2im
    # kernel_col = kernel.reshape(2, -1)  # 2 is number of filters
    #
    # prod = np.matmul(img_col, kernel_col.T) # for im2col and col2im methods
    # # prod = np.matmul(kernel_col, img_col)
    # col_to_prod = im2col.col2im(prod.T, (2, 1, 2, 2), 2, 2)
    # # col_to_prod = im2col.col2im_indices(prod.T, (2, 1, 2, 2), 2, 2, padding=0)
    # print col_to_prod
    # exit()

    # # example 3: multichannel
    # img = np.random.rand(16, 16, 3, 3)
    # kernel = np.random.rand(16, 16, 1, 1)
    # output = np.random.rand(4, 3, 5, 5)
    #
    # img_col = im2col.im2col(input_data=img, filter_h=1, filter_w=1, stride=1, pad=0)
    #
    # print img_col.shape
    # print kernel.reshape(16, -1).shape
    # exit()
    #
    # output = np.concatenate((output, np.zeros((4, 1, 5, 5))), axis=1)
    # output_col = output.reshape(4, -1).T
    #
    # kernel = np.concatenate((kernel, np.zeros((4, 1, 2, 2))), axis=1)
    # kernel_col = kernel.reshape(4, -1).T  # 4 is number of filter
    #
    # kernel_col = np.linalg.lstsq(img_col, output_col, rcond=None)[0]
    #
    # exit()
    # lamda = 1e-4
    # error = 1e-5
    #
    # for i in range(50):
    #     img_col = np.linalg.solve(
    #         np.dot(kernel_col, kernel_col.T) + lamda * np.eye(kernel_col.shape[0]),
    #         np.dot(kernel_col, output_col.T)).T
    #
    #     kernel_col = np.linalg.solve(
    #         img_col.T.dot(img_col) + lamda * np.eye(img_col.shape[1]),
    #         np.dot(img_col.T, output_col))
    #
    #     print np.linalg.norm(np.dot(img_col, kernel_col) - output_col)
    #     if np.linalg.norm(np.dot(img_col, kernel_col) - output_col) < error:
    #         break
    #
    # kernel = kernel_col.T.reshape(4, 4, 3, 3)
    # kernel = kernel[:, :3, ...]
    # img = im2col.col2im(col=img_col, input_shape=img.shape, filter_h=3, filter_w=3, stride=1, pad=2)
    # exit()

    # example 4: test ALS multichannel
    # img = np.random.randint(0, 7, (1, 2, 3, 3))
    # kernel = np.random.randint(0, 2, (1, 2, 3, 3))
    # output = np.random.rand(4, 3, 5, 5)
    #
    # img = np.array([[[[4, 5, 4], [2, 0, 4], [5, 1, 1]],
    #                  [[0, 1, 1], [4, 4, 3], [0, 3, 6]]]])
    # img_col = im2col.im2col(img, 3, 3, 1, 1)
    # # img_col = im2col.im2col_indices(img, 3, 3, 1, 1)
    # print 'original image'
    # print img
    #
    # kernel = np.array([[[[0, 1, 1], [0, 1, 1], [0, 0, 0]],
    #                     [[1, 0, 0], [1, 0, 0], [1, 0, 1]]]])
    # kernel_col = kernel.reshape(kernel.shape[0], -1).T
    # print kernel_col.shape
    #
    # output = np.array([[[[13, 16, 9],
    #                      [14, 23, 16],
    #                      [8, 10, 12]]]])
    # output_col = output.reshape(output.shape[0], -1).T
    #
    # prod = np.dot(img_col, kernel_col)
    # output_calc = prod.reshape(1,1,3,3)
    # print output_calc
    # # #
    # # exit()
    # lamda = 1e-4
    # error = 1e-5
    #
    # kernel_col = np.linalg.lstsq(img_col, output_col, rcond=None)[0]
    #
    # for i in range(1):
    #     img_col = np.linalg.solve(
    #         np.dot(kernel_col, kernel_col.T) + lamda * np.eye(kernel_col.shape[0]),
    #         np.dot(kernel_col, output_col.T)).T
    #
    #     kernel_col = np.linalg.solve(
    #         img_col.T.dot(img_col) + lamda * np.eye(img_col.shape[1]),
    #         np.dot(img_col.T, output_col))
    #
    #     print np.linalg.norm(np.dot(img_col, kernel_col) - output_col)
    #     if np.linalg.norm(np.dot(img_col, kernel_col) - output_col) < error:
    #         break
    #
    # kernel = kernel_col.T.reshape(kernel.shape)
    # # kernel = kernel[:, :3, ...]
    # img = im2col.col2im(col=img_col, input_shape=img.shape, filter_h=3,
    #                     filter_w=3, stride=1, padding=1)
    # img = img / [[4, 6, 4], [6, 9, 6], [4, 6, 4]]
    # print 'calc image'
    # print img
    # # exit()
    #
    # img_col = im2col.im2col(img, 3, 3, 1, 1)
    # prod = np.dot(img_col, kernel_col)
    # output = im2col.col2im(prod, (1, 1, 3, 3), 3, 3, 1, 0)
    # print 'new prod'
    # print output
    # exit()

    logs = []
    colors = []
    trace_names = []

    if args.plot_name is not None:
        visdom_live_plot = PlotLearning(
            './plots/cifar/', 10, plot_name=args.plot_name, env_name=args.env_name)
    else:
        visdom_live_plot = None

    start_time = time.time()
    print("\n\n > Teacher (Base Network) training ... ")
    net_type = 'Teacher'
    colors.append('orange')
    trace_names.extend(['Teacher Train', 'Teacher Test'])
    teacher_model = ConvNet(net_dataset=CIFAR10)
    teacher_model.cuda()
    optimizer = get_optimizer(teacher_model)
    scheduler = get_scheduler(optimizer)
    print teacher_model
    log_base, win_accuracy, win_loss = start_training(
        teacher_model, net_type, optimizer, scheduler, visdom_live_plot)
    logs.append(log_base)
    save_optimizer_scheduler(optimizer, scheduler, net_type)
    end_time = time.time()

    print 'time to train teacher network:',
    print end_time-start_time
    # wider student training from Net2Net
    print("\n\n > Wider Student training (Net2Net)... ")
    net_type = 'WideNet2Net'
    colors.append('blue')
    trace_names.extend(['Wider Net2Net Train', 'Wider Net2Net Test'])
    n2n_model_wider = copy.deepcopy(teacher_model)
    n2n_model_wider.load_state_dict(th.load(os.path.join(MODEL_PATH, args.plot_name + '_bestmodel.pt')))
    n2n_model_wider.wider('net2net', widening_factor=2)
    optimizer = get_optimizer(n2n_model_wider)
    scheduler = get_scheduler(optimizer)
    print n2n_model_wider
    log_net2net, win_accuracy, win_loss = start_training(
        n2n_model_wider, net_type, optimizer, scheduler,
        visdom_live_plot, win_accuracy, win_loss)
    logs.append(log_net2net)
    save_optimizer_scheduler(optimizer, scheduler, net_type)

    exit()
    # # wider model training from scratch
    # print("\n\n > Wider Network training (Wider Random Init)... ")
    # net_type = 'WideRandInit'
    # colors.append('green')
    # trace_names.extend(['Wider Random Train', 'Wider Random Test'])
    # wider_random_init_model = ConvNet(net_dataset=CIFAR10)
    # wider_random_init_model.define_wider(widening_factor=2)
    # # wider_random_init_model.apply(weights_init)
    # wider_random_init_model.cuda()
    # optimizer = get_optimizer(wider_random_init_model)
    # scheduler = get_scheduler(optimizer)
    # print wider_random_init_model
    # log_random_init, win_accuracy, win_loss = start_training(
    #     wider_random_init_model, net_type, optimizer, scheduler,
    #     visdom_live_plot, win_accuracy, win_loss)
    # logs.append(log_random_init)
    # save_optimizer_scheduler(optimizer, scheduler, net_type)

    # # wider student training from NetMorph
    # print("\n\n > Wider Student training (NetMorph)... ")
    # net_type = 'WideNetMorph'
    # colors.append('red')
    # trace_names.extend(['Wider NetMorph Train', 'Wider NetMorph Test'])
    # netmorph_model_wider = ConvNet(net_dataset=CIFAR10)
    # netmorph_model_wider.cuda()
    # netmorph_model_wider = copy.deepcopy(teacher_model)
    # netmorph_model_wider.load_state_dict(
    #     th.load(os.path.join(MODEL_PATH, args.plot_name + '_bestmodel.pt')))
    # netmorph_model_wider.wider('netmorph', widening_factor=2)
    # optimizer = get_optimizer(netmorph_model_wider)
    # scheduler = get_scheduler(optimizer)
    # print netmorph_model_wider
    # log_netmorph, win_accuracy, win_loss = start_training(
    #     netmorph_model_wider, net_type, optimizer, scheduler,
    #     visdom_live_plot, win_accuracy, win_loss)
    # logs.append(log_netmorph)
    # save_optimizer_scheduler(optimizer, scheduler, net_type)

    # # deeper model training from scratch
    # print("\n\n > Deeper Network training (Random Init)... ")
    # net_type = 'DeepRandomInit'
    # colors.append('green')
    # trace_names.extend(['Deeper Random Train', 'Deeper Random Test'])
    # deeper_random_init_model = ConvNet(net_dataset=CIFAR10)
    # deeper_random_init_model.define_deeper(deepening_factor=2)
    # deeper_random_init_model.cuda()
    # optimizer = get_optimizer(deeper_random_init_model)
    # scheduler = get_scheduler(optimizer)
    # print deeper_random_init_model
    # log_random_init, win_accuracy, win_loss = start_training(
    #     deeper_random_init_model, net_type, optimizer, scheduler,
    #     visdom_live_plot, win_accuracy, win_loss)
    # logs.append(log_random_init)

    # # Deeper student training from Net2Net
    # print("\n\n > Deeper Student training (Net2Net)... ")
    # net_type = 'DeeperNet2Net'
    # colors.append('blue')
    # trace_names.extend(['Deeper Net2Net Train', 'Deeper Net2Net Test'])
    # n2n_model_deeper = copy.deepcopy(teacher_model)
    # n2n_model_deeper.deeper('net2net')
    # n2n_model_deeper.cuda()
    # optimizer = get_optimizer(n2n_model_deeper)
    # scheduler = get_scheduler(optimizer)
    # print n2n_model_deeper
    # log_net2net, win_accuracy, win_loss = start_training(
    #     n2n_model_deeper, net_type, optimizer, scheduler,
    #     visdom_live_plot, win_accuracy, win_loss)
    # logs.append(log_net2net)

    # Deeper student training from NetMorph
    print("\n\n > Deeper Student training (NetMorph)... ")
    net_type = 'DeepNetMorph'
    colors.append('red')
    trace_names.extend(['Deeper NetMorph Train', 'Deeper NetMorph Test'])
    netmorph_model_deeper = copy.deepcopy(teacher_model)
    netmorph_model_deeper.deeper('netmorph')
    netmorph_model_deeper.cuda()
    optimizer = get_optimizer(netmorph_model_deeper)
    scheduler = get_scheduler(optimizer)
    print netmorph_model_deeper
    log_netmorph, win_accuracy, win_loss = start_training(
        netmorph_model_deeper, net_type, optimizer, scheduler,
        visdom_live_plot, win_accuracy, win_loss)
    logs.append(log_netmorph)

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

