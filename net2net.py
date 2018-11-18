import torch as th
import torch.nn as nn
import numpy as np

NOISE_RATIO = 1e-5
ERROR_TOLERANCE = 1e-3


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

def verify_TF(teacher_w1, teacher_b1, teacher_w2, student_w1, student_b1, student_w2):
    import scipy.signal
    inputs = np.random.rand(teacher_w1.shape[0] * 4, teacher_w1.shape[1] * 4, teacher_w1.shape[2])
    ori1 = np.zeros((teacher_w1.shape[0] * 4, teacher_w1.shape[1] * 4, teacher_w1.shape[3]))
    ori2 = np.zeros((teacher_w1.shape[0] * 4, teacher_w1.shape[1] * 4, teacher_w2.shape[3]))
    new1 = np.zeros((teacher_w1.shape[0] * 4, teacher_w1.shape[1] * 4, student_w1.shape[3]))
    new2 = np.zeros((teacher_w1.shape[0] * 4, teacher_w1.shape[1] * 4, student_w2.shape[3]))

    for i in range(teacher_w1.shape[3]):
        for j in range(inputs.shape[2]):
            if j == 0:
                tmp = scipy.signal.convolve2d(inputs[:, :, j], teacher_w1[:, :, j, i], mode='same')
            else:
                tmp += scipy.signal.convolve2d(inputs[:, :, j], teacher_w1[:, :, j, i], mode='same')
        ori1[:, :, i] = tmp + teacher_b1[i]

    for i in range(teacher_w2.shape[3]):
        for j in range(ori1.shape[2]):
            if j == 0:
                tmp = scipy.signal.convolve2d(ori1[:, :, j], teacher_w2[:, :, j, i], mode='same')
            else:
                tmp += scipy.signal.convolve2d(ori1[:, :, j], teacher_w2[:, :, j, i], mode='same')
        ori2[:, :, i] = tmp

    for i in range(student_w1.shape[3]):
        for j in range(inputs.shape[2]):
            if j == 0:
                tmp = scipy.signal.convolve2d(inputs[:, :, j], student_w1[:, :, j, i], mode='same')
            else:
                tmp += scipy.signal.convolve2d(inputs[:, :, j], student_w1[:, :, j, i], mode='same')
        new1[:, :, i] = tmp + student_b1[i]

    for i in range(student_w2.shape[3]):
        for j in range(new1.shape[2]):
            if j == 0:
                tmp = scipy.signal.convolve2d(new1[:, :, j], student_w2[:, :, j, i], mode='same')
            else:
                tmp += scipy.signal.convolve2d(new1[:, :, j], student_w2[:, :, j, i], mode='same')
        new2[:, :, i] = tmp

    err = np.abs(np.sum(ori2 - new2))

    assert err < ERROR_TOLERANCE, 'Verification failed: [ERROR] {}'.format(err)




def _wider_TF(teacher_w1, teacher_b1, teacher_w2, rand):

    replication_factor = np.bincount(rand)
    student_w1 = teacher_w1.copy()
    student_w2 = teacher_w2.copy()
    student_b1 = teacher_b1.copy()

    for i in range(len(rand)):
        teacher_index = rand[i]
        new_weight = teacher_w1[:, :, :, teacher_index]
        new_weight = new_weight[:, :, :, np.newaxis]
        student_w1 = np.concatenate((student_w1, new_weight), axis=3)
        student_b1 = np.append(student_b1, teacher_b1[teacher_index])

    for i in range(len(rand)):
        teacher_index = rand[i]
        factor = replication_factor[teacher_index] + 1
        assert factor > 1, 'Error in Net2Wider'
        new_weight = teacher_w2[:, :, teacher_index, :]*(1./factor)
        new_weight_re = new_weight[:, :, np.newaxis, :]
        student_w2 = np.concatenate((student_w2, new_weight_re), axis=2)
        student_w2[:, :, teacher_index, :] = new_weight

    # verify_TF(teacher_w1, teacher_b1, teacher_w2, student_w1, student_b1, student_w2)


def _wider_TH(w1, b1, w2, rand):

    tw1 = th.from_numpy(w1).type(th.DoubleTensor)
    tw2 = th.from_numpy(w2).type(th.DoubleTensor)
    tb1 = th.from_numpy(b1).type(th.DoubleTensor)

    sw1 = tw1.clone()
    sb1 = tb1.clone()
    sw2 = tw2.clone()


    replication_factor = np.bincount(rand)

    for i in range(rand.numel()):
        teacher_index = int(rand[i].item())
        new_weight = tw1.select(0, teacher_index)
        new_weight = new_weight.unsqueeze(0)
        sw1 = th.cat((sw1, new_weight), dim=0)
        new_bias = tb1[teacher_index].unsqueeze(0)
        sb1 = th.cat((sb1, new_bias))

    for i in range(rand.numel()):
        teacher_index = int(rand[i].item())
        factor_index = replication_factor[teacher_index] + 1
        assert factor_index > 1, 'Error in Net2Wider'
        new_weight = tw2.select(1, teacher_index) * (1. / factor_index)
        new_weight_re = new_weight.unsqueeze(1)
        sw2 = th.cat((sw2, new_weight_re), dim=1)
        sw2[:, teacher_index, :, :] = new_weight

    verify_TH(w1, b1, w2, sw1.numpy(), sb1.numpy(), sw2.numpy())


def wider(layer1, layer2, new_width, bnorm=None, out_size=None, noise=True, random_init=False, weight_norm=True):
    print 'Net2Net Widening... '
    w1 = layer1.weight.data
    w2 = layer2.weight.data
    b1 = layer1.bias.data
    b2 = layer2.bias.data

    if "Conv" in layer1.__class__.__name__ and ("Conv" in layer2.__class__.__name__ or "Linear" in layer2.__class__.__name__):

        # Convert Linear layers to Conv if linear layer follows target layer
        if "Conv" in layer1.__class__.__name__ and "Linear" in layer2.__class__.__name__:
            assert w2.size(1) % w1.size(0) == 0, "Linear units need to be multiple"
            if w1.dim() == 4:
                factor = int(np.sqrt(w2.size(1) // w1.size(0)))
                w2 = w2.view(
                    w2.size(0), w2.size(1) // factor ** 2, factor, factor)
        else:
            assert w1.size(0) == w2.size(1), "Module weights are not compatible"

        assert new_width > w1.size(0), "New size should be larger"

        nw1 = w1.clone()
        nb1 = b1.clone()
        nw2 = w2.clone()

        old_width = w1.size(0)

        if bnorm is not None:
            nrunning_mean = bnorm.running_mean.clone().resize_(new_width)
            nrunning_var = bnorm.running_var.clone().resize_(new_width)
            if bnorm.affine:
                nweight = bnorm.weight.data.clone().resize_(new_width)
                nbias = bnorm.bias.data.clone().resize_(new_width)

            nrunning_var.narrow(0, 0, old_width).copy_(bnorm.running_var)
            nrunning_mean.narrow(0, 0, old_width).copy_(bnorm.running_mean)
            if bnorm.affine:
                nweight.narrow(0, 0, old_width).copy_(bnorm.weight.data)
                nbias.narrow(0, 0, old_width).copy_(bnorm.bias.data)

        rand_ids = th.randint(low=0, high=w1.shape[0], size=((new_width - w1.shape[0]),))
        replication_factor = np.bincount(rand_ids)

        if isinstance(layer1, nn.Conv2d):
            new_current_layer = nn.Conv2d(
                out_channels=new_width, in_channels=layer1.in_channels,
                kernel_size=(3, 3), stride=1, padding=1)
        else:
            new_current_layer = nn.Linear(
                in_features=layer1.out_channels * layer1.kernel_size[0] * layer1.kernel_size[1],
                out_features=layer2.out_features)

        for i in range(rand_ids.numel()):
            teacher_index = int(rand_ids[i].item())
            new_weight = w1.select(0, teacher_index)
            new_weight = add_noise(new_weight, nw1)
            new_weight = new_weight.unsqueeze(0)
            nw1 = th.cat((nw1, new_weight), dim=0)

            new_bias = b1[teacher_index].unsqueeze(0)
            nb1 = th.cat((nb1, new_bias))

            if bnorm is not None:
                nrunning_mean[old_width + i] = bnorm.running_mean[teacher_index]
                nrunning_var[old_width + i] = bnorm.running_var[teacher_index]
                if bnorm.affine:
                    nweight[old_width + i] = bnorm.weight.data[teacher_index]
                    nbias[old_width + i] = bnorm.bias.data[teacher_index]
                bnorm.num_features = new_width

        new_current_layer.weight.data = nw1
        new_current_layer.bias.data = nb1
        layer1 = new_current_layer

        for i in range(rand_ids.numel()):
            teacher_index = int(rand_ids[i].item())
            factor_index = replication_factor[teacher_index] + 1
            assert factor_index > 1, 'Error in Net2Wider'
            new_weight = w2.select(1, teacher_index) * (1. / factor_index)
            new_weight_re = new_weight.unsqueeze(1)
            nw2 = th.cat((nw2, new_weight_re), dim=1)
            nw2[:, teacher_index, :, :] = new_weight

        if isinstance(layer2, nn.Conv2d):
            new_next_layer = nn.Conv2d(out_channels=layer2.out_channels,
                                       in_channels=new_width,
                                       kernel_size=(3, 3), stride=1, padding=1)
        else:
            new_next_layer = nn.Linear(
                in_features=layer1.out_channels * layer1.kernel_size[0] * layer1.kernel_size[1],
                out_features=layer2.out_features)

        if "Conv" in layer1.__class__.__name__ and "Linear" in layer2.__class__.__name__:
            new_next_layer.in_features = new_width * factor ** 2
            new_next_layer.weight.data = nw2.view(layer2.weight.size(0), new_width * factor ** 2)
        else:
            new_next_layer.weight.data = nw2

        new_next_layer.bias.data = b2
        layer2 = new_next_layer

        layer1.weight.data = nw1
        layer1.bias.data = nb1

        if bnorm is not None:
            bnorm.running_var = nrunning_var
            bnorm.running_mean = nrunning_mean
            if bnorm.affine:
                bnorm.weight.data = nweight
                bnorm.bias.data = nbias

        return layer1, layer2, bnorm


def deeper(layer, activation_fn=nn.ReLU(), bnorm=True, prefix=''):
    r""" Function preserving deeper operator adding a new layer on top of the
    given layer.

    Implemented based on Net2Net paper. If a new dense layer is being added, its
    weight matrix will be set to identity matrix. For convolutional layer, the
    center element of a input channel (in increasing sequence) is set to 1 and
    other to 0. This approach only works only for Relu activation function as it
    is idempotent.

    :param layer: Layer on top of which new layers will be added.
    :param activation_fn: Activation function to be used between the two layers.
     Default Relu
    :param bnorm: Add a batch normalisation layer between two
    convolutional/dense layers if True.

    :return: New layers to be added in the network.
    """

    print 'Net2Net Deeper...'
    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
        if isinstance(layer, nn.Linear):
            # Create new linear layer with input and output features equal to
            # output features of a dense layer on top of which a new dense layer
            # is being added.
            new_layer = th.nn.Linear(layer.out_features, layer.out_features)
            new_layer.weight.data = th.eye(layer.out_features)
            new_layer.bias.data = th.zeros(layer.out_features)

            if bnorm:
                new_num_features = layer.out_features
                new_bn_layer = nn.BatchNorm1d(num_features=new_num_features)
        else:
            new_filter_shape = layer.kernel_size
            new_num_channels = layer.out_channels
            # Create new convolutional layer with number of input and output
            # channels equal to number of output channel of the layer on top of
            # which new layer will be placed. The filter shape will be same. And
            # a padding of 1 is added to maintain previous output dimension.
            new_layer = th.nn.Conv2d(new_num_channels, new_num_channels,
                                     kernel_size=layer.kernel_size, padding=1)

            new_layer_weight = th.zeros(
                (new_num_channels, new_num_channels) + new_filter_shape)
            center = tuple(map(lambda x: int((x - 1) / 2), new_filter_shape))
            for i in range(new_num_channels):
                filter_weight = th.zeros((new_num_channels,) + new_filter_shape)
                index = (i,) + center
                filter_weight[index] = 1
                new_layer_weight[i, ...] = filter_weight

            new_layer_bias = th.zeros(new_num_channels)
            # Set new weight and bias for new convolutional layer
            new_layer.weight.data = new_layer_weight
            new_layer.bias.data = new_layer_bias

            # Set noise as initial weight and bias for all parameter values for
            # BN layer
            if bnorm:
                new_num_features = layer.out_channels
                new_bn_layer = nn.BatchNorm2d(num_features=new_num_features)

        new_bn_layer.weight.data = add_noise(
            th.ones(new_num_features).cuda(), th.Tensor([0, 1]))
        new_bn_layer.bias.data = add_noise(
            th.zeros(new_num_features).cuda(), th.Tensor([0, 1]))
        new_bn_layer.running_mean.data = add_noise(
            th.zeros(new_num_features).cuda(), th.Tensor([0, 1]))
        new_bn_layer.running_var.data = add_noise(
            th.ones(new_num_features).cuda(), th.Tensor([0, 1]))
    else:
        raise RuntimeError(
            "{} Module not supported".format(layer.__class__.__name__))

    seq_container = th.nn.Sequential().cuda()
    seq_container.add_module(prefix + '_conv', layer)
    if bnorm:
        seq_container.add_module(prefix + '_bnorm', new_bn_layer)
    if activation_fn is not None:
        seq_container.add_module(prefix + '_nonlin', nn.ReLU())
    seq_container.add_module(prefix + '_conv_new', new_layer)

    return seq_container

# def deeper(layer, activation_fn=nn.ReLU(), bnorm=True, prefix=''):
#     r""" Function preserving deeper operator adding a new layer on top of the
#     given layer.
#
#     Implemented based on Net2Net paper. If a new dense layer is being added, its
#     weight matrix will be set to identity matrix. For convolutional layer, the
#     center element of a input channel (in increasing sequence) is set to 1 and
#     other to 0. This approach only works only for Relu activation function as it
#     is idempotent.
#
#     :param layer: Layer on top of which new layers will be added.
#     :param activation_fn: Activation function to be used between the two layers.
#      Default Relu
#     :param bnorm: Add a batch normalisation layer between two
#     convolutional/dense layers if True.
#
#     :return: New layers to be added in the network.
#     """
#
#     if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
#         if isinstance(layer, nn.Linear):
#             # Create new linear layer with input and output features equal to
#             # output features of a dense layer on top of which a new dense layer
#             # is being added.
#             new_layer = th.nn.Linear(layer.out_features, layer.out_features)
#             new_layer.weight.data = th.eye(layer.out_features)
#             new_layer.bias.data = th.zeros(layer.out_features)
#
#             if bnorm:
#                 new_num_features = layer.out_features
#                 new_bn_layer = nn.BatchNorm1d(num_features=new_num_features)
#         else:
#             new_filter_shape = layer.kernel_size
#             new_num_channels = layer.out_channels
#             # Create new convolutional layer with number of input and output
#             # channels equal to number of output channel of the layer on top of
#             # which new layer will be placed. The filter shape will be same. And
#             # a padding of 1 is added to maintain previous output dimension.
#             new_layer = th.nn.Conv2d(new_num_channels, new_num_channels,
#                                      kernel_size=layer.kernel_size, padding=1)
#
#             new_layer_weight = th.zeros(
#                 (new_num_channels, new_num_channels) + new_filter_shape)
#             center = tuple(map(lambda x: int((x - 1) / 2), new_filter_shape))
#             for i in range(new_num_channels):
#                 filter_weight = th.zeros((new_num_channels,) + new_filter_shape)
#                 index = (i,) + center
#                 filter_weight[index] = 1
#                 new_layer_weight[i, ...] = filter_weight
#
#             new_layer_bias = th.zeros(new_num_channels)
#             # Set new weight and bias for new convolutional layer
#             new_layer.weight.data = new_layer_weight
#             new_layer.bias.data = new_layer_bias
#
#             # Set noise as initial weight and bias for all parameter values for
#             # BN layer
#             if bnorm:
#                 new_num_features = layer.out_channels
#                 new_bn_layer = nn.BatchNorm2d(num_features=new_num_features)
#
#         new_bn_layer.weight.data = add_noise(
#             th.ones(new_num_features).cuda(), th.Tensor([0, 1]))
#         new_bn_layer.bias.data = add_noise(
#             th.zeros(new_num_features).cuda(), th.Tensor([0, 1]))
#         new_bn_layer.running_mean.data = add_noise(
#             th.zeros(new_num_features).cuda(), th.Tensor([0, 1]))
#         new_bn_layer.running_var.data = add_noise(
#             th.ones(new_num_features).cuda(), th.Tensor([0, 1]))
#     else:
#         raise RuntimeError(
#             "{} Module not supported".format(layer.__class__.__name__))
#
#     # TODO: Check code to add new layers
#     seq_container = th.nn.Sequential()
#     seq_container.add_module(prefix + '_conv', layer)
#     if bnorm:
#         seq_container.add_module(prefix + '_bnorm', new_bn_layer)
#     if activation_fn is not None:
#         seq_container.add_module(prefix + '_nonlin', nn.ReLU())
#     seq_container.add_module(prefix + '_conv_new', new_layer)
#
#     return seq_container


if __name__ == '__main__':

    # with file('test_data_w1.txt', 'w') as test_data_w1:
    #     test_data_w1.write('# Array Shape: {0}\n'.format(w1.shape))
    #     for data_slice in w1:
    #         for slice2 in data_slice:
    #             np.savetxt(test_data_w1, slice2, fmt='%-4.8f')
    #             test_data_w1.write('# BBBB\n')
    #         test_data_w1.write('# AAAA\n')
    #
    # w1_file = np.loadtxt('test_data_w1.txt').reshape(3, 3, 2, 3)

    # # for tensor flow verification
    # new_width = 384
    # w1 = np.random.rand(3, 3, 128, 256)
    # b1 = np.random.rand(256)
    # w2 = np.random.rand(3, 3, 256, 512)
    # rand = np.random.randint(w1.shape[3], size=(new_width - w1.shape[3]))
    # # print(rand)
    # # rand = th.randint(low=0, high=tw1.shape[0], size=((new_width - tw1.shape[0]),))
    # _wider_TF(w1, b1, w2, rand)
    #
    # # for torch verification
    # w1 = w1.reshape(256, 128, 3, 3)
    # w2 = w2.reshape(512, 256, 3, 3)
    # _wider_TH(w1, b1, w2, th.from_numpy(rand))
    # for wider operation verification
    _test_wider_operation()
