import torch as th
import torch.nn as nn
import numpy as np
import random
import sys

sys.path.append('./')
from utils import add_noise

ERROR_TOLERANCE = 1e-3


def _test_wider_operation():
    ip_channel_1 = 3
    op_channel_1 = 16
    ip_channel_2 = op_channel_1
    op_channel_2 = op_channel_1 * 2
    new_width = 32
    kernel_size = 3

    teacher_conv1 = nn.Conv2d(ip_channel_1, op_channel_1, kernel_size).cuda()
    teacher_bn1 = nn.BatchNorm2d(op_channel_1).cuda()
    teacher_conv2 = nn.Conv2d(ip_channel_2, op_channel_2, kernel_size).cuda()

    tw1 = teacher_conv1.weight.data.to('cpu') # or .cpu() like below
    tw2 = teacher_conv2.weight.data.cpu()
    tbn1 = teacher_bn1.weight.data.cpu()

    student_conv1, student_conv2, student_bnorm = wider(
        teacher_conv1, teacher_conv2, new_width, teacher_bn1)

    sw1 = student_conv1.weight.data.cpu()
    sw2 = student_conv2.weight.data.cpu()
    sb1 = student_bnorm.weight.data.cpu()

    verify_weights(tw1.numpy(), tbn1.numpy(), tw2.numpy(),
                   sw1.numpy(), sb1.numpy(), sw2.numpy())


def verify_weights(teacher_w1, teacher_b1, teacher_w2,
                   student_w1, student_b1, student_w2):
    import scipy.signal
    test_input = np.random.rand(teacher_w1.shape[1], teacher_w1.shape[3] * 4, teacher_w1.shape[2] * 4)
    ori1 = np.zeros((teacher_w1.shape[0], teacher_w1.shape[3] * 4, teacher_w1.shape[2] * 4))
    ori2 = np.zeros((teacher_w2.shape[0], teacher_w1.shape[3] * 4, teacher_w1.shape[2] * 4))
    new1 = np.zeros((student_w1.shape[0], teacher_w1.shape[3] * 4, teacher_w1.shape[2] * 4))
    new2 = np.zeros((student_w2.shape[0], teacher_w1.shape[3] * 4, teacher_w1.shape[2] * 4))

    for i in range(teacher_w1.shape[0]):
        for j in range(test_input.shape[0]):
            if j == 0:
                tmp = scipy.signal.convolve2d(test_input[j], teacher_w1[i, j], mode='same')
            else:
                tmp += scipy.signal.convolve2d(test_input[j], teacher_w1[i, j], mode='same')
        ori1[i] = tmp + teacher_b1[i]
    for i in range(teacher_w2.shape[0]):
        for j in range(ori1.shape[0]):
            if j == 0:
                tmp = scipy.signal.convolve2d(ori1[j], teacher_w2[i, j], mode='same')
            else:
                tmp += scipy.signal.convolve2d(ori1[j], teacher_w2[i, j], mode='same')
        ori2[i] = tmp

    for i in range(student_w1.shape[0]):
        for j in range(test_input.shape[0]):
            if j == 0:
                tmp = scipy.signal.convolve2d(test_input[j], student_w1[i, j], mode='same')
            else:
                tmp += scipy.signal.convolve2d(test_input[j], student_w1[i, j], mode='same')
        new1[i] = tmp + student_b1[i]
    for i in range(student_w2.shape[0]):
        for j in range(new1.shape[0]):
            if j == 0:
                tmp = scipy.signal.convolve2d(new1[j], student_w2[i, j], mode='same')
            else:
                tmp += scipy.signal.convolve2d(new1[j], student_w2[i, j], mode='same')
        new2[i] = tmp

    err = np.abs(np.sum(ori2 - new2))

    print 'the err val is:' + str(err)
    assert err < ERROR_TOLERANCE, 'Verification failed: [ERROR] {}'.format(err)


def wider(layer1, layer2, new_width, bnorm=None):

    print 'Net2Net Widening... '
    w1 = layer1.weight.data
    w2 = layer2.weight.data
    b1 = layer1.bias.data
    b2 = layer2.bias.data

    if isinstance(layer1, nn.Conv2d) and (isinstance(layer2, nn.Conv2d)
                                          or isinstance(layer2, nn.Linear)):

        # Convert Linear layers to Conv if linear layer follows target layer
        if isinstance(layer1, nn.Conv2d) and isinstance(layer2, nn.Linear):
	    print w2.shape
	    print w1.shape
	    print w2.size(1)
	    print w1.size(0)
            assert w2.size(1) % w1.size(0) == 0, 'Linear units need to be multiple'
            if w1.dim() == 4:
                kernel_size = int(np.sqrt(w2.size(1) // w1.size(0)))
		print kernel_size
		exit()
                w2 = w2.view(
                    w2.size(0), w2.size(1) // kernel_size ** 2, kernel_size, kernel_size)
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

        if isinstance(layer1, nn.Conv2d):
            new_current_layer = nn.Conv2d(
                out_channels=new_width, in_channels=layer1.in_channels,
                kernel_size=(3, 3), stride=1, padding=1)
        else:
            new_current_layer = nn.Linear(
                in_features=layer1.out_channels * layer1.kernel_size[0] * layer1.kernel_size[1],
                out_features=layer2.out_features)

        rand_ids = th.tensor(random.sample(range(w1.shape[0]), new_width - w1.shape[0]))
        replication_factor = np.bincount(rand_ids)

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

        new_current_layer.weight.data = nw1
        new_current_layer.bias.data = nb1
        layer1 = new_current_layer

        # Copy the weights from input channel of next layer and append it after
        # dividing the selected filter by replication factor.
        for i in range(rand_ids.numel()):
            teacher_index = int(rand_ids[i].item())
            factor = replication_factor[teacher_index] + 1
            assert factor > 1, 'Error in Net2Wider'
            # Calculate new weight according to replication factor
            new_weight = w2.select(1, teacher_index) * (1. / factor)
            # Append the new weight increasing its input channel
            new_weight_re = new_weight.unsqueeze(1)
            nw2 = th.cat((nw2, new_weight_re), dim=1)
            # Assign the calculated new weight to replicated filter
            nw2[:, teacher_index, :, :] = new_weight

        if isinstance(layer2, nn.Conv2d):
            new_next_layer = nn.Conv2d(out_channels=layer2.out_channels,
                                       in_channels=new_width,
                                       kernel_size=(3, 3), stride=1, padding=1)
            new_next_layer.weight.data = nw2
        else:
            new_next_layer = nn.Linear(
                in_features=layer1.out_channels * layer1.kernel_size[0] * layer1.kernel_size[1],
                out_features=layer2.out_features)
            # Convert the 4D tensor to 2D tensor for linear layer i.e. reverse
            # the earlier effect when linear layer was converted to
            # convolutional layer.
            new_next_layer.weight.data = nw2.view(
                layer2.weight.size(0), new_width * kernel_size ** 2)

        # Set the bias for new next layer as previous bias for next layer
        new_next_layer.bias.data = b2
        layer2 = new_next_layer

        if bnorm is not None:
            bnorm.num_features = new_width
            bnorm.running_var = nrunning_var
            bnorm.running_mean = nrunning_mean
            if bnorm.affine:
                bnorm.weight.data = nweight
                bnorm.bias.data = nbias

        return layer1, layer2, bnorm


def deeper(layer, activation_fn=nn.ReLU(), bnorm=True, prefix='', filters=16):
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
    :param filters: Number of filters of filters being deepened

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
            new_kernel_shape = layer.kernel_size
            new_num_channels = filters
            # Create new convolutional layer with number of input and output
            # channels equal to number of output channel of the layer on top of
            # which new layer will be placed. The filter shape will be same. And
            # a padding of 1 is added to maintain previous output dimension.
            new_layer = th.nn.Conv2d(new_num_channels, new_num_channels,
                                     kernel_size=layer.kernel_size, padding=1)

            new_layer_weight = th.zeros(
                (new_num_channels, new_num_channels) + new_kernel_shape)
            center = tuple(map(lambda x: int((x - 1) / 2), new_kernel_shape))
            for i in range(new_num_channels):
                filter_weight = th.zeros((new_num_channels,) + new_kernel_shape)
                index = (i,) + center
                filter_weight[index] = 1
                new_layer_weight[i, ...] = filter_weight

            new_layer_bias = th.zeros(new_num_channels)
            # Set new weight and bias for new convolutional layer
            # new_layer.weight.data = new_layer_weight
            new_layer.weight.data = add_noise(new_layer_weight.cuda(), layer.weight.data)
            new_layer.bias.data = new_layer_bias

            # Set noise as initial weight and bias for all parameter values for
            # BN layer
            if bnorm:
                new_num_features = layer.out_channels
                new_bn_layer = nn.BatchNorm2d(num_features=new_num_features)

        if bnorm:
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
    # if activation_fn is not None:
    #     seq_container.add_module(prefix + '_nonlin', nn.ReLU())
    seq_container.add_module(prefix + '_conv_new', new_layer)

    return seq_container


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

    # for wider operation verification
    _test_wider_operation()
