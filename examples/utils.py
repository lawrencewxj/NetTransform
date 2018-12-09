"""
Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
"""
from requests.exceptions import ConnectionError
from visdom import Visdom

import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pylab as plt
import numpy as np
import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.init as init

label_names = [
    'airplane',
    'automobile',
    'bird',
    'cat',
    'deer',
    'dog',
    'frog',
    'horse',
    'ship',
    'truck'
]

def plot_images(images, cls_true, cls_pred=None):
    """
    Adapted from https://github.com/Hvass-Labs/TensorFlow-Tutorials/
    """
    fig, axes = plt.subplots(3, 3)

    for i, ax in enumerate(axes.flat):
        # plot img
        ax.imshow(images[i, :, :, :], interpolation='spline16')

        # show true & predicted classes
        cls_true_name = label_names[cls_true[i]]
        if cls_pred is None:
            xlabel = "{0} ({1})".format(cls_true_name, cls_true[i])
        else:
            cls_pred_name = label_names[cls_pred[i]]
            xlabel = "True: {0}\nPred: {1}".format(
                cls_true_name, cls_pred_name
            )
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])

    plt.show()

# TODO: when executed from bash, image not showing
def show_sample_image():
    pass

    # def imshow(img):
    #     img = img / 2 + 0.5     # unnormalize
    #     npimg = img.numpy()
    #     plt.imshow(np.transpose(npimg, (1, 2, 0)))
    #
    # # get some random training images
    # dataiter = iter(train_loader)
    # images, labels = dataiter.next()
    #
    # # show images
    # imshow(make_grid(images))
    # # print labels
    # print(' '.join('%5s' % CIFAR10.CLASSES[labels[j]] for j in range(4)))

def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


_, term_width = os.popen('stty size', 'r').read().split()
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time

def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

class NLL_loss_instance(torch.nn.NLLLoss):

    def __init__(self, ratio):
        super(NLL_loss_instance, self).__init__(None, True)
        self.ratio = ratio

    def forward(self, x, y, ratio=None):
        if ratio is not None:
            self.ratio = ratio
        num_inst = x.size(0)
        num_hns = int(self.ratio * num_inst)
        x_ = x.clone()
        for idx, label in enumerate(y.data):
            x_.data[idx, label] = 0.0
        loss_incs = -x_.sum(1)
        _, idxs = loss_incs.topk(num_hns)
        x_hn = x.index_select(0, idxs)
        y_hn = y.index_select(0, idxs)
        return torch.nn.functional.nll_loss(x_hn, y_hn)


class PlotLearning(object):

    def __init__(self, save_path, num_classes, plot_name='', env_name='experiments'):
        self.DEFAULT_PORT = 9898
        self.DEFAULT_HOSTNAME = 'http://130.83.143.241'
        self.accuracy = []
        self.val_accuracy = []
        self.losses = []
        self.val_losses = []
        self.save_path_loss = os.path.join(save_path, plot_name + 'loss_plot.png')
        self.save_path_accu = os.path.join(save_path, plot_name + 'accu_plot.png')
        self.init_loss = -np.log(1.0 / num_classes)
        self.viz = Visdom(port=self.DEFAULT_PORT, server=self.DEFAULT_HOSTNAME, env = env_name)
        self.plot_name = plot_name

    def plot(self, logs):
        self.accuracy.append(logs.get('acc'))
        self.val_accuracy.append(logs.get('val_acc'))
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))

        best_val_acc = max(self.val_accuracy)
        best_train_acc = max(self.accuracy)
        best_val_epoch = self.val_accuracy.index(best_val_acc)
        best_train_epoch = self.accuracy.index(best_train_acc)

        plt.figure(1)
        plt.gca().cla()
        plt.ylim(0, 1)
        plt.plot(self.accuracy, label='train')
        plt.plot(self.val_accuracy, label='valid')
        plt.title("best_val@{0:}-{1:.2f}, best_train@{2:}-{3:.2f}".format(
            best_val_epoch, best_val_acc, best_train_epoch, best_train_acc))
        plt.legend()
        plt.savefig(self.save_path_accu)

        best_val_loss = min(self.val_losses)
        best_train_loss = min(self.losses)
        best_val_epoch = self.val_losses.index(best_val_loss)
        best_train_epoch = self.losses.index(best_train_loss)

        plt.figure(2)
        plt.gca().cla()
        plt.ylim(0, self.init_loss)
        plt.plot(self.losses, label='train')
        plt.plot(self.val_losses, label='valid')
        plt.title("best_val@{0:}-{1:.2f}, best_train@{2:}-{3:.2f}".format(
            best_val_epoch, best_val_loss, best_train_epoch, best_train_loss))
        plt.legend()
        plt.savefig(self.save_path_loss)

    def plot_logs(self, logs, trace_names, colors):
        accuracy_traces = []
        loss_traces = []

        trace_names_index = 0
        for i in range(len(logs)):
            trace1 = dict(x=logs[i]['epoch'], y=logs[i]['train_accuracy'],
                          mode="lines", type='custom',
                          line={'color': colors[i], 'dash': 'dash'},
                          name=trace_names[trace_names_index])
            accuracy_traces.append(trace1)
            trace_names_index += 1
            trace2 = dict(x=logs[i]['epoch'], y=logs[i]['test_accuracy'],
                          mode="lines", type='custom',
                          line={'color': colors[i], 'shape': 'spline',
                                'smoothing': 1.3},
                          name=trace_names[trace_names_index])
            accuracy_traces.append(trace2)
            trace_names_index += 1

        with open(os.path.join(
                './logs', 'accuracy_' + self.plot_name +'.json'), 'w') as accuracy_file:
            json.dump(accuracy_traces, accuracy_file)

        layout = dict(title="Accuracy Vs Epoch - " + self.plot_name,
                      xaxis={'title': 'Epochs'}, yaxis={'title': 'Accuracy'})
        self.viz._send({'data': accuracy_traces,
                        'layout': layout, 'win': 'Accuracy_' + self.plot_name})

        trace_names_index = 0
        for i in range(len(logs)):
            trace1 = dict(x=logs[i]['epoch'], y=logs[i]['train_loss'],
                          mode="lines", type='custom',
                          line={'color': colors[i], 'dash': 'dash'},
                          name=trace_names[trace_names_index])
            loss_traces.append(trace1)
            trace_names_index += 1
            trace2 = dict(x=logs[i]['epoch'], y=logs[i]['test_loss'],
                          mode="lines", type='custom',
                          line={'color': colors[i], 'shape': 'spline',
                                'smoothing': 1.3},
                          name=trace_names[trace_names_index])
            loss_traces.append(trace2)
            trace_names_index += 1

        with open(
                os.path.join('./logs', 'loss_' + self.plot_name + '.json'),
                'w') as loss_file:
            json.dump(loss_traces, loss_file)

        layout = dict(title="Loss Vs Epoch - " + self.plot_name,
                      xaxis={'title': 'Epochs'}, yaxis={'title': 'Loss'})
        self.viz._send({'data': loss_traces,
                        'layout': layout, 'win': 'Loss_' + self.plot_name})

    def plot_live_logs(self, accuracy_traces, loss_traces):

        try:
            layout = dict(title="Accuracy Vs Epoch - " + self.plot_name,
                          xaxis={'title': 'Epochs'},
                          yaxis={'title': 'Accuracy'})
            self.viz._send({'data': accuracy_traces, 'layout': layout,
                            'win': 'Accuracy' + self.plot_name})

            layout = dict(title="Loss Vs Epoch - " + self.plot_name,
                          xaxis={'title': 'Epochs'}, yaxis={'title': 'Loss'})
            self.viz._send({'data': loss_traces, 'layout': layout,
                            'win': 'Loss_' + self.plot_name})
        except ConnectionError:
            print('Connection error...')
            time.sleep(5)

    def plot_saved_data(self, plot_data):
        layout = dict(title="Accuracy Vs Epoch - " + self.plot_name,
                      xaxis={'title': 'Epochs'}, yaxis={'title': 'Accuracy'})
        self.viz._send({'data': plot_data,
                        'layout': layout, 'win': 'Accuracy_check'})


if __name__ == '__main__':

    filename = './logs/accuracy_run10.json'

    visdom_plot_final = PlotLearning(
        './plots/cifar/', 10, plot_name='run10', env_name='check_plot')

    with open(filename, 'r') as read_file:
        json_data = json.load(read_file)

    visdom_plot_final.plot_saved_data(plot_data=json_data)

