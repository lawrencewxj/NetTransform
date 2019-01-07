import torch as th
import torch.nn as nn
import numpy as np
import sys
import im2col

sys.path.append('./')
from utils import add_noise

ERROR_TOLERANCE = 1e-2


np.set_printoptions(threshold=np.nan)


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

    verify_weights_wider(tw1.numpy(), tb1.numpy(), tw2.numpy(),
                   sw1.numpy(), sb1.numpy(), sw2.numpy())


def _test_deeper_operation():
    ip_channel_1 = 3
    op_channel_1 = 128
    new_output_filters = 192
    kernel_size = 3

    teacher_conv1 = nn.Conv2d(ip_channel_1, op_channel_1, kernel_size, padding=1)
    student_conv1 = deeper(teacher_conv1, nn.ReLU, bnorm=False, prefix='l1',
                           filters=new_output_filters)

    verify_weights_deeper(teacher_conv1, student_conv1)


def verify_weights_deeper(teacher_w1, student_w1):
    inputs = th.from_numpy(np.random.rand(1, 3, 32, 32)).float()

    output = teacher_w1(inputs)
    output = output.detach()
    new_output = student_w1(inputs)
    new_output = new_output.detach()

    print output.shape
    print new_output.shape

    err = np.abs(np.sum((new_output - output).detach().numpy()))

    assert err < ERROR_TOLERANCE, 'Verification failed: [ERROR] {}'.format(err)


def verify_weights_wider(teacher_w1, teacher_b1, teacher_w2,
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


def general_netmorph(parent_filter_wt):
    pass


def practical_netmorph(parent_filter_wt):
    pass


def decompose_filter(parent_filter_wt, filters=16):
    lamda = 1e-4
    error = 1e-7

    c1 = parent_filter_wt.shape[1]
    c2 = parent_filter_wt.shape[0]
    k = parent_filter_wt.shape[2]
    k1 = k
    k2 = k
    k_expanded = k1 + k2 - 1
    pad_zero = nn.ZeroPad2d((k_expanded - k) / 2)

    new_weight = pad_zero(parent_filter_wt)
    # new_weight = add_noise(new_weight, new_weight)
    new_weight = new_weight.cpu().numpy()
    # output is the original parent filter which is generated by
    # convolving an image (=img_col) by filter (=kernel)
    # output_col is the 2D representation of an output generated after matrix
    # multiplication
    # output = new_weight

    # wt = parent_filter_wt.cpu().numpy()
    # mean = wt.mean()
    # std = wt.std()
    # var = wt.var()
    output = np.concatenate(
        (new_weight, np.zeros((c2, c2 - c1, k_expanded, k_expanded))), axis=1)

    # NOISE_RATIO = 1e-5
    # noise_range = NOISE_RATIO * np.ptp(parent_filter_wt.flatten())
    # noise = np.random.uniform(-noise_range, noise_range, size=output.shape)
    # output = output + noise

    output_col = output.reshape(c2, -1).T
    # output_col2 = np.random.normal(0, 1e-2, size=output_col.shape)

    # print np.linalg.norm(output_col2 - output_col)
    # exit()
    # Below 2 lines can be removed
    # kernel is equivalent to filter f1 which will convolve image (=img_col)
    # kernel = np.random.normal(0, 1e-2, size=(c2, c, k1, k1))
    kernel = np.random.choice(output.flatten(), size=(c2, filters, k1, k1))
    kernel_col = kernel.reshape(c2, -1).T

    # img is the f2 filter treated as image to be convolved by f1(=kernel)
    # img_col is the 2D representation of a filter for matrix multiplication
    # img = np.random.normal(0, 1e-2, size=(c2, c, k2, k2))
    img = np.random.choice(output.flatten(), size=(c2, filters, k2, k2))
    # # exit()
    img_col = im2col.im2col(img, k1, k1, stride=1, padding=2)
    # img_col = np.random.normal(
    #     0, 1e-2, size=(k_expanded * k_expanded * c2, k1 * k1 * c))
    # img_col_original = img_col.copy()
    # kernel_col = np.linalg.lstsq(img_col, output_col, rcond=None)[0]

    # print kernel_col.shape
    # exit()
    for i in range(10):
        img_col = np.linalg.solve(
            np.dot(kernel_col, kernel_col.T) + lamda * np.eye(
                kernel_col.shape[0]),
            np.dot(kernel_col, output_col.T)).T

        kernel_col = np.linalg.solve(
            img_col.T.dot(img_col) + lamda * np.eye(img_col.shape[1]),
            np.dot(img_col.T, output_col))

        print np.linalg.norm(np.dot(img_col, kernel_col) - output_col)
        if np.linalg.norm(np.dot(img_col, kernel_col) - output_col) < error:
            break

    print 'before converting to original image after calculating prod: ',
    new_prod = np.dot(img_col, kernel_col)
    print np.linalg.norm(new_prod - output_col)

    kernel = kernel_col.T.reshape(filters, c2, k1, k1)
    kernel = kernel[:, :c1, ...]

    img_calculated = im2col.col2im(col=img_col, input_shape=(c2, filters, k2, k2),
                                   filter_h=k1, filter_w=k1,
                                   padding=k_expanded - k)
    # img_calculated = im2col.recover_input(
    #     input=img_col, kernel_size=k1, stride=1, outshape=(c2, c, k2, k2))
    img_calculated = img_calculated / 9  # because original matrix elements are added 9 times , for double padding
    # img_calculated = img_calculated/[[1, 2, 1], [2, 4, 2], [1, 2, 1]] for zero padding
    # img = (img / ([[4, 6, 4], [6, 9, 6], [4, 6, 4]]))/ for single padding
    # print 'image_col error: ',
    # print np.linalg.norm((img_col - img_col_original))
    # print 'image error: ',
    # print np.linalg.norm((img_calculated - img))
    #
    img_col2 = im2col.im2col(img_calculated, 3, 3, 1, 2)
    print 'after converting, product error = ',
    print np.linalg.norm(np.dot(img_col2, kernel_col) - output_col)
    # img = im2col.recover_input(input=img_col, kernel_size=k1, stride=1,
    #                            outshape=(c2, c, k2, k2))
    # exit()
    # *************************************************************
    # img = np.random.normal(0, 1e-2, size=(4, 4, 3, 3))
    # original_img = img.copy()
    # img_col = im2col.im2col(img, 3, 3, 1, 2)
    # original_img_col = img_col.copy()
    # # print img_col
    # # img_col = np.random.randint(0, 4, size=(100, 36))
    # # print img_col[0, 0]
    # # kernel_col = np.random.randint(0, 2, size=(36, 4))
    # output_col = np.random.normal(0, 1e-2, size=(100, 4))
    #
    # kernel_col = np.linalg.lstsq(img_col, output_col, rcond=None)[0]
    #
    # for i in range(100):
    #     img_col = np.linalg.solve(
    #         np.dot(kernel_col, kernel_col.T) + lamda * np.eye(
    #             kernel_col.shape[0]),
    #         np.dot(kernel_col, output_col.T)).T
    #
    #     kernel_col = np.linalg.solve(
    #         img_col.T.dot(img_col) + lamda * np.eye(img_col.shape[1]),
    #         np.dot(img_col.T, output_col))
    #
    #     # print np.linalg.norm(np.dot(img_col, kernel_col) - output_col)
    #     if np.linalg.norm(np.dot(img_col, kernel_col) - output_col) < error:
    #         break
    #
    # print 'before converting after calcultating',
    # new_prod = np.dot(img_col, kernel_col)
    # print np.linalg.norm(new_prod - output_col)
    #
    # print np.linalg.norm(img_col - original_img_col)
    # img = im2col.col2im(img_col, (4, 4, 3, 3), 3, 3, padding=2)
    # img = img / 9
    #
    # print np.linalg.norm(img - original_img)
    # img_col2 = im2col.im2col(img, 3, 3, padding=2)
    # new_prod = np.dot(img_col2, kernel_col)
    # print np.linalg.norm(new_prod - output_col)
    # exit()
    # *************************************************************

    return kernel, img_calculated


def deeper(layer, activation_fn=nn.ReLU(), bnorm=True, prefix='', filters=16):
    print 'NetMorph Deeper ...'

    if isinstance(layer, nn.Linear) or isinstance(layer, nn.Conv2d):
        if isinstance(layer, nn.Linear):
            pass
        else:
            teacher_weight = layer.weight.data
            teacher_bias = layer.bias.data

            f1, f2 = decompose_filter(teacher_weight, filters)
            # f1, f2 = practical_netmorph(teacher_weight)

            # new_num_channels = layer.out_channels
            new_layer1 = th.nn.Conv2d(f1.shape[1], f1.shape[0],
                                      kernel_size=layer.kernel_size, padding=1)
            new_layer2 = th.nn.Conv2d(f2.shape[1], f2.shape[0],
                                      kernel_size=layer.kernel_size, padding=1)

            new_layer1.weight.data = th.from_numpy(f1).float()
            new_layer2.weight.data = th.from_numpy(f2).float()

            new_layer1.bias.data = th.zeros(new_layer1.out_channels)
            new_layer2.bias.data = th.zeros(new_layer2.out_channels)

            if bnorm:
                new_num_features = new_layer1.out_channels
                new_bn_layer = nn.BatchNorm2d(num_features=new_num_features)

                new_bn_layer.weight.data = add_noise(
                    th.ones(new_num_features).cuda(), th.Tensor([0, 1]))
                new_bn_layer.bias.data = add_noise(
                    th.zeros(new_num_features).cuda(), th.Tensor([0, 1]))
                new_bn_layer.running_mean.data = add_noise(
                    th.zeros(new_num_features).cuda(), th.Tensor([0, 1]))
                new_bn_layer.running_var.data = add_noise(
                    th.ones(new_num_features).cuda(), th.Tensor([0, 1]))

    seq_container = th.nn.Sequential().cuda()
    seq_container.add_module(prefix + '_conv', new_layer1)
    if bnorm:
        seq_container.add_module(prefix + '_bnorm', new_bn_layer)
    # if activation_fn is not None:
    #     seq_container.add_module(prefix + '_nonlin', nn.ReLU())
    seq_container.add_module(prefix + '_conv_new', new_layer2)

    return seq_container


if __name__ == '__main__':

    # for wider operation verification
    # _test_wider_operation()

    # for deeper operation verification
    _test_deeper_operation()