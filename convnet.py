import torch.nn as nn
import torch.nn.functional as F

from netmorph import wider
from net2net import deeper

BASE_WIDTH = 4


class CIFAR10(object):
    INPUT_CHANNELS = 3
    NUM_OUTPUT_CLASSES = 10
    CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog',
               'horse', 'ship', 'truck']


class MNIST(object):
    INPUT_CHANNELS = 1
    OUTPUT_CLASSES = 10


class ConvNet(nn.Module):
    def __init__(self, net_dataset):
        super(ConvNet, self).__init__()
        self.net_dataset = net_dataset
        self.conv1 = nn.Conv2d(out_channels=BASE_WIDTH,
                               in_channels=self.net_dataset.INPUT_CHANNELS,
                               kernel_size=(3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=self.conv1.out_channels)
        self.pool1 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)

        self.conv2 = nn.Conv2d(out_channels=self.conv1.out_channels * 2,
                               in_channels=self.conv1.out_channels,
                               kernel_size=(3, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=self.conv2.out_channels)
        self.pool2 = nn.MaxPool2d(kernel_size=(3, 3), stride=2)

        self.conv3 = nn.Conv2d(out_channels=self.conv2.out_channels * 2,
                               in_channels=self.conv2.out_channels,
                               kernel_size=(3, 3), stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=self.conv3.out_channels)
        self.pool3 = nn.AvgPool2d(5, 1)

        self.fc1 = nn.Linear(
            out_features=self.net_dataset.NUM_OUTPUT_CLASSES,
            in_features=self.conv3.out_channels * self.conv3.kernel_size[0] *
                        self.conv3.kernel_size[1])

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        try:
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            x = self.pool3(F.relu(self.bn3(self.conv3(x))))
            x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
            x = F.relu(self.fc1(x))
            return x
        except RuntimeError:
            print(x.size())

    def netmorph_wider(self, widening_factor):
        r""" Widen the Convolutional net by given widening factor

        :param widening_factor: factor to increase the width of all layers in
         convolutional net except input channel of first convolutional layer
         and output channel of output layer

        :return:
        """

        self.conv1, self.conv2, self.bn1 = wider(
            self.conv1, self.conv2, self.conv1.out_channels * widening_factor,
            self.bn1)
        self.conv2, self.conv3, self.bn2 = wider(
            self.conv2, self.conv3, self.conv2.out_channels * widening_factor,
            self.bn2)
        self.conv3, self.fc1, self.bn3 = wider(
            self.conv3, self.fc1, self.conv3.out_channels * widening_factor,
            self.bn3)

        # TODO: check for 2 adjacent FC layers
        # self.fc1, self.fc2, _ = wider(self.fc1, self.fc2,
        #                               self.fc1.out_channel * widening_factor,
        #                               self.bn4)

    def define_wider(self, widening_factor):
        self.conv1 = nn.Conv2d(
            out_channels=self.conv1.out_channels * widening_factor,
            in_channels=self.net_dataset.INPUT_CHANNELS,
            kernel_size=(3, 3), stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features=self.conv1.out_channels)
        self.conv2 = nn.Conv2d(
            out_channels=self.conv2.out_channels * widening_factor,
            in_channels=self.conv2.in_channels * widening_factor,
            kernel_size=(3, 3), stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features=self.conv2.out_channels)
        self.conv3 = nn.Conv2d(
            out_channels=self.conv3.out_channels * widening_factor,
            in_channels=self.conv3.in_channels * widening_factor,
            kernel_size=(3, 3), stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(num_features=self.conv3.out_channels)
        self.fc1 = nn.Linear(
            in_features=self.conv3.out_channels * self.conv3.kernel_size[0] * self.conv3.kernel_size[1],
            out_features=self.net_dataset.NUM_OUTPUT_CLASSES)

    def net2net_deeper(self):
        s = deeper(self.conv1, F.relu, bnorm=True)
