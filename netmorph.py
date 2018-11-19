import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

NOISE_RATIO = 1e-5
ERROR_TOLERANCE = 1e-2


def add_noise(weights, other_weights):
    noise_range = NOISE_RATIO * np.ptp(other_weights.flatten())
    noise = th.Tensor(weights.shape).uniform_(
        -noise_range / 2.0, noise_range / 2.0).cuda()

    return th.add(noise, weights)


def _test_wider_operation():
    ip_channel_1 = 3
    op_channel_1 = 128
    ip_channel_2 = op_channel_1
    op_channel_2 = op_channel_1 * 2
    new_width = 192
    kernel_size = 3

    teacher_conv1 = nn.Conv2d(ip_channel_1, op_channel_1, kernel_size).cuda()
    teacher_bn1 = nn.BatchNorm2d(op_channel_1).cuda()
    teacher_conv2 = nn.Conv2d(ip_channel_2, op_channel_2, 3).cuda()

    tw1 = teacher_conv1.weight.data.to('cpu') # or .cpu() like below
    tw2 = teacher_conv2.weight.data.cpu()
    tb1 = teacher_bn1.weight.data.cpu()

    student_conv1, student_conv2, student_bnorm = wider(
        teacher_conv1, teacher_conv2, new_width, teacher_bn1)
    # student_conv1, student_conv2, student_bnorm = wider_net2net(
    #     teacher_conv1, teacher_conv2, new_width, teacher_bn1)

    sw1 = student_conv1.weight.data.cpu()
    sw2 = student_conv2.weight.data.cpu()
    sb1 = student_bnorm.weight.data.cpu()

    verify_weights(tw1.numpy(), tb1.numpy(), tw2.numpy(),
                   sw1.numpy(), sb1.numpy(), sw2.numpy())


def verify_weights(teacher_w1, teacher_b1, teacher_w2,
                   student_w1, student_b1, student_w2):
    import scipy.signal
    inputs = np.random.rand(teacher_w1.shape[1], teacher_w1.shape[3] * 4, teacher_w1.shape[2] * 4)
    ori1 = np.zeros((teacher_w1.shape[0], teacher_w1.shape[3] * 4, teacher_w1.shape[2] * 4))
    ori2 = np.zeros((teacher_w2.shape[0], teacher_w1.shape[3] * 4, teacher_w1.shape[2] * 4))
    new1 = np.zeros((student_w1.shape[0], teacher_w1.shape[3] * 4, teacher_w1.shape[2] * 4))
    new2 = np.zeros((student_w2.shape[0], teacher_w1.shape[3] * 4, teacher_w1.shape[2] * 4))

    for i in range(teacher_w1.shape[0]):
        for j in range(inputs.shape[0]):
            if j == 0:
                tmp = scipy.signal.convolve2d(inputs[j, :, :], teacher_w1[i, j, :, :], mode='same')
            else:
                tmp += scipy.signal.convolve2d(inputs[j, :, :], teacher_w1[i, j, :, :], mode='same')
        ori1[i, :, :] = tmp + teacher_b1[i]
    for i in range(teacher_w2.shape[0]):
        for j in range(ori1.shape[0]):
            if j == 0:
                tmp = scipy.signal.convolve2d(ori1[j, :, :], teacher_w2[i, j, :, :], mode='same')
            else:
                tmp += scipy.signal.convolve2d(ori1[j, :, :], teacher_w2[i, j, :, :], mode='same')
        ori2[i, :, :] = tmp

    for i in range(student_w1.shape[0]):
        for j in range(inputs.shape[0]):
            if j == 0:
                tmp = scipy.signal.convolve2d(inputs[j, :, :], student_w1[i, j, :, :], mode='same')
            else:
                tmp += scipy.signal.convolve2d(inputs[j, :, :], student_w1[i, j, :, :], mode='same')
        new1[i, :, :] = tmp + student_b1[i]
    for i in range(student_w2.shape[0]):
        for j in range(new1.shape[0]):
            if j == 0:
                tmp = scipy.signal.convolve2d(new1[j, :, :], student_w2[i, j, :, :], mode='same')
            else:
                tmp += scipy.signal.convolve2d(new1[j, :, :], student_w2[i, j, :, :], mode='same')
        new2[i, :, :] = tmp

    err = np.abs(np.sum(ori2 - new2))

    assert err < ERROR_TOLERANCE, 'Verification failed: [ERROR] {}'.format(err)


def wider(layer1, layer2, new_width, bnorm=None):
    r""" Widens the layers in the network.

    Implemented according to NetMorph Widening operation. The next adjacent
    layer in the network also needs to be be widened due to increase in the
    width of previous layer.

    :param layer1: The layer to be widened
    :param layer2: The next adjacent layer to be widened
    :param new_width: Width of the new layer (output channels/features of first
    layer and input channels/features of next layer.
    :param bnorm: BN layer to be widened if provided.
    :return: widened layers
    """

    print 'NetMorph Widening... '
    if (isinstance(layer1, nn.Conv2d) or isinstance(layer1, nn.Linear)) and (
            isinstance(layer2, nn.Conv2d) or isinstance(layer2, nn.Linear)):

        teacher_w1 = layer1.weight.data
        teacher_b1 = layer1.bias.data
        teacher_w2 = layer2.weight.data
        teacher_b2 = layer2.bias.data

        assert new_width > teacher_w1.size(0), "New size should be larger"

        # Widening output channels/features of first layer
        # Randomly select weight from the first teacher layer and corresponding
        # bias and add it to first student layer. Add noise to newly created
        # student layer.
        student_w1 = teacher_w1.clone()
        student_b1 = teacher_b1.clone()

        rand_ids = th.randint(low=0, high=teacher_w1.shape[0],
                              size=((new_width - teacher_w1.shape[0]),))

        for i in range(rand_ids.numel()):
            teacher_index = int(rand_ids[i].item())
            new_weight = teacher_w1[teacher_index, ...]
            new_weight.unsqueeze_(0)
            student_w1 = th.cat((student_w1, new_weight), dim=0)
            new_bias = teacher_b1[teacher_index]
            new_bias.unsqueeze_(0)
            student_b1 = th.cat((student_b1, new_bias))

        if isinstance(layer1, nn.Conv2d):
            new_current_layer = nn.Conv2d(
                out_channels=new_width, in_channels=layer1.in_channels,
                kernel_size=(3, 3), stride=1, padding=1)
        else:
            new_current_layer = nn.Linear(
                in_features=layer1.out_channels * layer1.kernel_size[0] * layer1.kernel_size[1],
                out_features=layer2.out_features)

        new_current_layer.weight.data = add_noise(student_w1, teacher_w1)
        new_current_layer.bias.data = add_noise(student_b1, teacher_b1)
        layer1 = new_current_layer

        # Widening input channels/features of second layer. Copy the weights
        # from teacher layer and only add noise to additional filter
        # channels/features in student layer. The student layer will have same
        # bias as teacher.
        new_weight = th.zeros(teacher_w2.shape).cuda()
        noise = add_noise(new_weight, teacher_w2)

        student_w2 = th.cat((teacher_w2, noise), dim=1)

        if isinstance(layer2, nn.Conv2d):
            new_next_layer = nn.Conv2d(out_channels=layer2.out_channels,
                                       in_channels=new_width,
                                       kernel_size=(3, 3), stride=1, padding=1)
        else:
            new_next_layer = nn.Linear(
                in_features=layer1.out_channels * layer1.kernel_size[0] * layer1.kernel_size[1],
                out_features=layer2.out_features)

        new_next_layer.weight.data = student_w2
        new_next_layer.bias.data = teacher_b2
        layer2 = new_next_layer

    # Widening batch normalisation layer if provided. Only add noise to
    # additional features for all 4 parameters in the layer i.e. mean, variance,
    # weight and bias.
    if bnorm is not None:
        n_add = new_width - bnorm.num_features

        # get current parameter values
        bn_weights = bnorm.weight.data
        bn_bias = bnorm.bias.data
        bn_running_mean = bnorm.running_mean.data
        bn_running_var = bnorm.running_var.data

        # set noise for all parameter values
        weight_noise = add_noise(th.ones(n_add).cuda(), th.Tensor([0, 1]))
        bias_noise = add_noise(th.zeros(n_add).cuda(), th.Tensor([0, 1]))
        running_mean_noise = add_noise(th.zeros(n_add).cuda(),
                                       th.Tensor([0, 1]))
        running_var_noise = add_noise(th.ones(n_add).cuda(), th.Tensor([0, 1]))

        # append noise to current parameter values to widen
        new_bn_weights = th.cat((bn_weights, weight_noise))
        new_bn_bias = th.cat((bn_bias, bias_noise))
        new_bn_running_mean = th.cat((bn_running_mean, running_mean_noise))
        new_bn_running_var = th.cat((bn_running_var, running_var_noise))

        # assign new parameter values for new BN layer
        new_bn_layer = nn.BatchNorm2d(num_features=bnorm.num_features + n_add)
        new_bn_layer.weight.data = new_bn_weights
        new_bn_layer.bias.data = new_bn_bias
        new_bn_layer.running_mean.data = new_bn_running_mean
        new_bn_layer.running_var.data = new_bn_running_var

        bnorm = new_bn_layer

    return layer1, layer2, bnorm


def deeper(layer, activation_fn=nn.ReLU(), bnorm=True, prefix=''):
    print 'NetMorph Deeper ...'

    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
        if isinstance(layer, nn.Linear):
            pass
        else:
            teacher_weight = layer.weight.data
            teacher_bias = layer.bias.data
            k = teacher_weight.size(2)
            k1 = k
            k2 = k
            kc = k1 + k2 - 1
            pad = nn.ConstantPad2d((kc - k) / 2, 0)
            new_weight = pad(teacher_weight)
            f1 = th.rand(teacher_weight.size(0), teacher_weight.size(1), k1, k1)
            f2 = th.rand(teacher_weight.size(0), teacher_weight.size(0), k2, k2)

            for i in xrange(50):
                d = 0


    seq_container = th.nn.Sequential().cuda()
    seq_container.add_module(prefix + '_conv', layer)
    if bnorm:
        seq_container.add_module(prefix + '_bnorm', new_bn_layer)
    if activation_fn is not None:
        seq_container.add_module(prefix + '_nonlin', nn.ReLU())
    seq_container.add_module(prefix + '_conv_new', new_layer)

    return seq_container

if __name__ == '__main__':

    # for wider operation verification
    _test_wider_operation()
