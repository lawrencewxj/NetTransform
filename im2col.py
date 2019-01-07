import numpy as np


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    # First figure out what the size of the output should be
    N, C, H, W = x_shape
    assert (H + 2 * padding - field_height) % stride == 0
    assert (W + 2 * padding - field_height) % stride == 0
    out_height = (H + 2 * padding - field_height) / stride + 1
    out_width = (W + 2 * padding - field_width) / stride + 1

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)

    return k, i, j


def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    """
    An implementation of im2col based on some fancy indexing
    """

    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p, p), (p, p)), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding,
                                 stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)

    return cols


def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1,
                   stride=1):
    """ An implementation of col2im based on fancy indexing and np.add.at """

    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                                 stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    if padding == 0:
        return x_padded

    return x_padded[:, :, padding:-padding, padding:-padding]


def im2col(input_data, filter_h, filter_w, stride=1, padding=0):
    """
    Parameters
    ----------
    input_data: input data consisting of a 4-dimensional array of (number of data, channel, height, width)
    filter_h: Filter height
    filter_w: Filter width
    stride: stride
    padding: Padding
    Returns
    -------
    Col : 2 dimensional allocation
    """

    N, C, H, W = input_data.shape
    out_h = (H + 2 * padding - filter_h) // stride + 1
    out_w = (W + 2 * padding - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (padding, padding), (padding, padding)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col


def col2im(col, input_shape, filter_h, filter_w, stride=1, padding=0):
    """
    Parameters
    ----------
    col :
    input_shape: Input data shape (example: (10, 1, 28, 28))
    filter_h :
    filter_w
    stride
    padding
    Returns
    -------
    """
    N, C, H, W = input_shape
    out_h = (H + 2 * padding - filter_h) // stride + 1
    out_w = (W + 2 * padding - filter_w) // stride + 1

    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
    img = np.zeros((N, C, H + 2 * padding + stride - 1, W + 2 * padding + stride - 1))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            img[:, :, y:y_max:stride, x:x_max:stride] += col[:, :, y, x, :, :]

    return img[:, :, padding:H + padding, padding:W + padding]


def recover_input(input, kernel_size, stride, outshape):
    """
    :param input: it is of the shape (height, width)
    :param kernel_size: it is the kernel shape we want
    :param stride:
    :param outshape: the shape of the output
    :return:
    """

    print 'input shape',
    print input.shape
    H, W = input.shape
    batch, ch, h, w = outshape
    original_input = np.zeros(outshape)
    first_row_index = np.arange(0, w, kernel_size)
    first_col_index = np.arange(0, h, kernel_size)

    patches_row = int((w - kernel_size) / stride) + 1
    rowend_index = kernel_size - (w - first_row_index[-1])
    colend_index = kernel_size - (h - first_col_index[-1])

    if first_row_index[-1] + kernel_size > w:
        first_row_index[-1] = first_row_index[-1] - (first_row_index[-1] + kernel_size - 1 - (w - 1))
    if first_col_index[-1] + kernel_size > h:
        first_col_index[-1] = first_col_index[-1] - (first_col_index[-1] + kernel_size - 1 - (h - 1))

    for k in range(batch):
        for i in range(len(first_col_index)):
            for j in range(len(first_row_index)):
                w_index = first_row_index[j] + i * patches_row + k * (int((h - kernel_size) / stride) + 1) * (int((w - kernel_size) / stride) + 1)

                if i != len(first_col_index) - 1 and j != len(first_row_index) - 1:
                    original_input[k, first_row_index[j]: first_row_index[j] + kernel_size, first_col_index[i]:  first_col_index[i]+kernel_size, :] = input[w_index, :].reshape(kernel_size, kernel_size, -1)
                elif i == len(first_col_index) - 1 and j != len(first_row_index) - 1:
                    original_input[k, first_col_index[-1] + colend_index:, first_row_index[i]:first_row_index[i] + kernel_size, :] = input[w_index, :].reshape(kernel_size, kernel_size, -1)[rowend_index:, :, :]
                elif i != len(first_col_index) - 1 and j == len(first_row_index) - 1:
                    original_input[k, first_col_index[i]:first_col_index[i] + kernel_size, first_row_index[-1] + rowend_index:, :] = input[w_index, :].reshape(kernel_size, kernel_size, -1)[:, colend_index:, :]
                else:
                    original_input[k, :, first_col_index[-1] + colend_index:, first_row_index[-1] + rowend_index:] = input[w_index, :].reshape(-1, kernel_size, kernel_size)[:, rowend_index:, colend_index:]

    return original_input