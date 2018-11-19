import unittest
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from netmorph import wider, deeper
# from net2net import wider, deeper

BASE_WIDTH = 8
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(out_channels=BASE_WIDTH,
                               in_channels=3,
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
        self.pool3 = nn.AvgPool2d(kernel_size=5, stride=1)

        self.fc1 = nn.Linear(
            out_features=10,
            in_features=self.conv3.out_channels * self.conv3.kernel_size[0] *
                        self.conv3.kernel_size[1])

    def forward(self, x):
        try:
            print x.shape
            x = self.pool1(F.relu(self.bn1(self.conv1(x))))
            print x.shape
            x = self.pool2(F.relu(self.bn2(self.conv2(x))))
            print x.shape
            x = self.pool3(F.relu(self.bn3(self.conv3(x))))
            print x.shape
            x = x.view(-1, x.size(1) * x.size(2) * x.size(3))
            print x.shape
            x = F.relu(self.fc1(x))
            return x
        except RuntimeError:
            print 'error during forward'
            print(x.size())


class TestOperators(unittest.TestCase):
    def _create_net(self):
        return Net()

    def test_wider(self):
        net = self._create_net()
        net.cuda()
        inp = th.autograd.Variable(th.rand(5, 3, 32, 32).cuda())

        # print 'before widening'
        # print net
        net.eval()
        out = net(inp)
        net.conv1, net.conv2, net.bn1 = wider(net.conv1, net.conv2, net.conv1.out_channels * 2, net.bn1)
        net.conv2, net.conv3, net.bn2 = wider(net.conv2, net.conv3, net.conv2.out_channels * 2, net.bn2)
        net.conv3, net.fc1, net.bn3 = wider(net.conv3, net.fc1, net.conv3.out_channels * 2, net.bn3)

        # print 'after widening'
        # print net
        net.eval()
        nout = net(inp)

        print th.abs((out - nout).sum().data).item() # [0]
        assert th.abs((out - nout).sum().data).item() < 1e-5
        # assert nout.size(0) == 32 and nout.size(1) == 10



    def dtest_deeper(self):
        net = self._create_net()
        inp = th.autograd.Variable(th.rand(5, 3, 32, 32))
        # print net
        net.eval()
        out = net(inp)

        s = deeper(net._modules['conv1'], nn.ReLU, bnorm=True)
        net._modules['conv1'] = s

        s2 = deeper(net._modules['conv2'], nn.ReLU, bnorm=True)
        net._modules['conv2'] = s2

        # s3 = deeper(net._modules['fc1'], nn.ReLU, bnorm_flag=True, weight_norm=False, noise=False)
        # net._modules['fc1'] = s3
        # print net
        # net.eval()
        # nout = net(inp)
        #
        # assert th.abs((out - nout).sum().data).item() < 1e-1

        net.eval()
        nout = net(inp)

        print th.abs((out - nout).sum().data).item()  # [0]
        assert th.abs((out - nout).sum().data).item() < 1e-5, "New layer changes values by {}".format(th.abs(out - nout).sum().data[0])

if __name__ == '__main__':
    unittest.main()
