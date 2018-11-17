import torch as th
import torch.nn as nn
import numpy as np

NOISE_RATIO = 1e-4
ERROR_TOLERANCE = 1e-5


def add_noise(weights, other_weights):
    noise_range = NOISE_RATIO * np.ptp(other_weights.flatten())
    noise = th.Tensor(weights.shape).uniform_(
        -noise_range / 2.0, noise_range / 2.0).cuda()

    return th.add(noise, weights)


def verify_TH(teacher_w1, teacher_b1, teacher_w2, student_w1, student_b1, student_w2):
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


def _wider_TH(w1, b1, w2, rand):

    tw1 = th.from_numpy(w1)
    tw2 = th.from_numpy(w2)
    tb1 = th.from_numpy(b1)

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


def wider_net2net(m1, m2, new_width, bnorm=None, noise=True):

    w1 = m1.weight.data
    w2 = m2.weight.data
    b1 = m1.bias.data

    if "Conv" in m1.__class__.__name__ and ("Conv" in m2.__class__.__name__ or "Linear" in m2.__class__.__name__):

        # Convert Linear layers to Conv if linear layer follows target layer
        if "Conv" in m1.__class__.__name__ and "Linear" in m2.__class__.__name__:
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

        rand_ids = th.randint(low=0, high=w1.shape[0],
                              size=((new_width - w1.shape[0]),))
        replication_factor = np.bincount(rand_ids)

        for i in range(rand_ids.numel()):
            teacher_index = int(rand_ids[i].item())
            new_weight = w1.select(0, teacher_index)
            if i > 0:
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

        for i in range(rand_ids.numel()):
            teacher_index = int(rand_ids[i].item())
            factor_index = replication_factor[teacher_index] + 1
            assert factor_index > 1, 'Error in Net2Wider'
            new_weight = w2.select(1, teacher_index) * (1. / factor_index)
            new_weight_re = new_weight.unsqueeze(1)
            nw2 = th.cat((nw2, new_weight_re), dim=1)
            nw2[:, teacher_index, :, :] = new_weight

        if "Conv" in m1.__class__.__name__ and "Linear" in m2.__class__.__name__:
            m2.in_features = new_width * factor ** 2
            m2.weight.data = nw2.view(m2.weight.size(0), new_width * factor ** 2)
        else:
            m2.weight.data = nw2

        m1.weight.data = nw1
        m1.bias.data = nb1

        if bnorm is not None:
            bnorm.running_var = nrunning_var
            bnorm.running_mean = nrunning_mean
            if bnorm.affine:
                bnorm.weight.data = nweight
                bnorm.bias.data = nbias

        return m1, m2, bnorm


if __name__ == '__main__':

    # for wider operation verification
    new_width = 384
    w1 = np.random.rand(256, 128, 3, 3)
    b1 = np.random.rand(256)
    w2 = np.random.rand(512, 256, 3, 3)
    rand = np.random.randint(w1.shape[0], size=(new_width - w1.shape[3]))

    _wider_TH(w1, b1, w2, th.from_numpy(rand))

