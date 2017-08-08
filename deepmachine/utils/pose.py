from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import menpo.io as mio
from menpo.image import Image
from menpo.shape import PointCloud

from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.contrib.framework.python.ops import variables
from tensorflow.python.training import optimizer as tf_optimizer
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import variables as tf_variables

from menpo.transform import Translation, Scale
from menpo.shape import PointCloud
from scipy.sparse import csr_matrix

slim = tf.contrib.slim

bodypose_matrix = csr_matrix((16, 16))

bodypose_matrix[0, 1] = 1
bodypose_matrix[1, 2] = 1
bodypose_matrix[2, 6] = 1
bodypose_matrix[6, 3] = 1
bodypose_matrix[3, 4] = 1
bodypose_matrix[4, 5] = 1
bodypose_matrix[6, 7] = 1
bodypose_matrix[7, 8] = 1
bodypose_matrix[8, 9] = 1
bodypose_matrix[10, 11] = 1
bodypose_matrix[11, 12] = 1
bodypose_matrix[12, 7] = 1
bodypose_matrix[7, 13] = 1
bodypose_matrix[13, 14] = 1
bodypose_matrix[14, 15] = 1

bodypose_matrix[1, 0] = 1
bodypose_matrix[2, 1] = 1
bodypose_matrix[6, 2] = 1
bodypose_matrix[3, 6] = 1
bodypose_matrix[4, 3] = 1
bodypose_matrix[5, 4] = 1
bodypose_matrix[7, 6] = 1
bodypose_matrix[8, 7] = 1
bodypose_matrix[9, 8] = 1
bodypose_matrix[11, 10] = 1
bodypose_matrix[12, 11] = 1
bodypose_matrix[7, 12] = 1
bodypose_matrix[13, 7] = 1
bodypose_matrix[14, 13] = 1
bodypose_matrix[15, 14] = 1


def generate_heatmap(logits, num_classes):
    """Generates a coloured heatmap from the keypoint logits.

    Args:
        features: A `Tensor` of dimensions [num_batch, height, width, FLAGS.n_landmarks + 1].
    """

    keypoint_colours = np.array(
        [
            plt.cm.spectral(x)
            for x in np.linspace(0, 1, num_classes + 1)
        ])[..., :3].astype(np.float32)

    prediction = tf.nn.softmax(logits)
    heatmap = tf.matmul(tf.reshape(
        prediction, (-1, num_classes + 1)), keypoint_colours)
    heatmap = tf.reshape(heatmap, (tf.shape(prediction)[0],
                                   tf.shape(prediction)[1],
                                   tf.shape(prediction)[2], 3))
    return heatmap


def generate_landmarks(keypoints):
    is_background = tf.equal(keypoints, 0)
    ones = tf.to_float(tf.ones_like(is_background))
    zeros = tf.to_float(tf.zeros_like(is_background))

    return tf.where(is_background, zeros, ones) * 255


def project_landmarks_to_shape_model(landmarks):
    final = []

    for lms in landmarks:
        lms = PointCloud(lms)
        similarity = AlignmentSimilarity(pca.global_transform.source, lms)
        projected_target = similarity.pseudoinverse().apply(lms)
        target = pca.model.reconstruct(projected_target)
        target = similarity.apply(target)
        final.append(target.points)

    return np.array(final).astype(np.float32)


def caffe_preprocess(image):
    VGG_MEAN = np.array([102.9801, 115.9465, 122.7717])

    # RGB -> BGR
    pixels = image.pixels[[2, 1, 0]]
    # Subtract VGG training mean across all channels
    pixels = pixels - VGG_MEAN.reshape([3, 1, 1])
    pixels = pixels.astype(np.float32, copy=False)
    return pixels


def rescale_image(image, stride_width=64):
    # make sure smallest size is 600 pixels wide & dimensions are (k * stride_width) + 1
    height, width = image.shape

    # Taken from 'szross'
    scale_up = 625. / min(height, width)
    scale_cap = 961. / max(height, width)
    scale_up = min(scale_up, scale_cap)
    new_height = stride_width * round((height * scale_up) / stride_width) + 1
    new_width = stride_width * round((width * scale_up) / stride_width) + 1
    image, tr = image.resize([new_height, new_width], return_transform=True)
    image.inverse_tr = tr
    return image


def frankotchellappa(dzdx, dzdy):
    from numpy.fft import ifftshift, fft2, ifft2
    rows, cols = dzdx.shape
    # The following sets up matrices specifying frequencies in the x and y
    # directions corresponding to the Fourier transforms of the gradient
    # data.  They range from -0.5 cycles/pixel to + 0.5 cycles/pixel.
    # The scaling of this is irrelevant as long as it represents a full
    # circle domain. This is functionally equivalent to any constant * pi
    pi_over_2 = np.pi / 2.0
    row_grid = np.linspace(-pi_over_2, pi_over_2, rows)
    col_grid = np.linspace(-pi_over_2, pi_over_2, cols)
    wy, wx = np.meshgrid(row_grid, col_grid, indexing='ij')

    # Quadrant shift to put zero frequency at the appropriate edge
    wx = ifftshift(wx)
    wy = ifftshift(wy)

    # Fourier transforms of gradients
    DZDX = fft2(dzdx)
    DZDY = fft2(dzdy)

    # Integrate in the frequency domain by phase shifting by pi/2 and
    # weighting the Fourier coefficients by their frequencies in x and y and
    # then dividing by the squared frequency
    denom = (wx ** 2 + wy ** 2)
    Z = (-1j * wx * DZDX - 1j * wy * DZDY) / denom
    Z = np.nan_to_num(Z)
    return np.real(ifft2(Z))


def create_train_op(
        total_loss,
        optimizer,
        global_step=None,
        update_ops=None,
        variables_to_train=None,
        clip_gradient_norm=0,
        iter_step=1,
        summarize_gradients=False,
        gate_gradients=tf_optimizer.Optimizer.GATE_OP,
        aggregation_method=None,
        colocate_gradients_with_ops=False,
        gradient_multipliers=None):
    """Creates an `Operation` that evaluates the gradients and returns the loss.
    Args:
    total_loss: A `Tensor` representing the total loss.
    optimizer: A tf.Optimizer to use for computing the gradients.
    global_step: A `Tensor` representing the global step variable. If left as
      `None`, then slim.variables.global_step() is used.
    update_ops: an optional list of updates to execute. Note that the update_ops
      that are used are the union of those update_ops passed to the function and
      the value of slim.ops.GetUpdateOps(). Therefore, if `update_ops` is None,
      then the value of slim.ops.GetUpdateOps() is still used.
    variables_to_train: an optional list of variables to train. If None, it will
      default to all tf.trainable_variables().
    clip_gradient_norm: If greater than 0 then the gradients would be clipped
      by it.
    iter_step: accumulate gradients across `iter_step` batches.
    summarize_gradients: Whether or not add summaries for each gradient.
    gate_gradients: How to gate the computation of gradients. See tf.Optimizer.
    aggregation_method: Specifies the method used to combine gradient terms.
      Valid values are defined in the class `AggregationMethod`.
    colocate_gradients_with_ops: Whether or not to try colocating the gradients
      with the ops that generated them.
    gradient_multipliers: A dictionary of either `Variables` or `Variable` op
      names to the coefficient by which the associated gradient should be
      scaled.
    Returns:
    A `Tensor` that when evaluated, computes the gradients and returns the total
      loss value.
    """
    if global_step is None:
        global_step = variables.get_or_create_global_step()

    # Update ops use GraphKeys.UPDATE_OPS collection if update_ops is None.
    global_update_ops = set(ops.get_collection(ops.GraphKeys.UPDATE_OPS))
    if update_ops is None:
        update_ops = global_update_ops
    else:
        update_ops = set(update_ops)

    if not global_update_ops.issubset(update_ops):
        logging.warning('update_ops in create_train_op does not contain all the '
                        ' update_ops in GraphKeys.UPDATE_OPS')

    # Make sure update_ops are computed before total_loss.
    if update_ops:
        with ops.control_dependencies(update_ops):
            barrier = control_flow_ops.no_op(name='update_barrier')
    total_loss = control_flow_ops.with_dependencies([barrier], total_loss)

    if variables_to_train is None:
        # Default to tf.trainable_variables()
        variables_to_train = tf_variables.trainable_variables()
    else:
        # Make sure that variables_to_train are in tf.trainable_variables()
        for v in variables_to_train:
            assert v in tf_variables.trainable_variables()

    assert variables_to_train

    # Create the gradients. Note that apply_gradients adds the gradient
    # computation to the current graph.
    single_grads = optimizer.compute_gradients(
        total_loss, variables_to_train, gate_gradients=gate_gradients,
        aggregation_method=aggregation_method,
        colocate_gradients_with_ops=colocate_gradients_with_ops)

    accum_grads = [tf.Variable(tf.zeros_like(g), trainable=False)
                   for (g, _) in single_grads]
    zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_grads]

    accum_ops = [a.assign_add(g)
                 for a, (g, _) in zip(accum_grads, single_grads)]
    grads = [(a / iter_step, v)
             for a, (_, v) in zip(accum_grads, single_grads)]

    def train_step_fn(sess, train_op, global_step, train_step_kwargs):
        sess.run(zero_ops)

        for i in range(iter_step):
            sess.run(accum_ops)

        return slim.learning.train_step(sess, train_op, global_step, train_step_kwargs)

    # Scale gradients.
    if gradient_multipliers:
        with ops.name_scope('multiply_grads'):
            grads = multiply_gradients(grads, gradient_multipliers)

    # Clip gradients.
    if clip_gradient_norm > 0:
        with ops.name_scope('clip_grads'):
            grads = clip_gradient_norms(grads, clip_gradient_norm)

    # Summarize gradients.
    if summarize_gradients:
        with ops.name_scope('summarize_grads'):
            slim.learning.add_gradients_summaries(grads)

    # Create gradient updates.
    grad_updates = optimizer.apply_gradients(grads, global_step=global_step)

    with ops.name_scope('train_op'):
        # Make sure total_loss is valid.
        total_loss = array_ops.check_numerics(total_loss,
                                              'LossTensor is inf or nan')

    # Ensure the train_tensor computes grad_updates.
    return control_flow_ops.with_dependencies([grad_updates], total_loss), train_step_fn


jaw_indices = np.arange(0, 17)
lbrow_indices = np.arange(17, 22)
rbrow_indices = np.arange(22, 27)
upper_nose_indices = np.arange(27, 31)
lower_nose_indices = np.arange(31, 36)
leye_indices = np.arange(36, 42)
reye_indices = np.arange(42, 48)
outer_mouth_indices = np.arange(48, 60)
inner_mouth_indices = np.arange(60, 68)

parts_68 = (jaw_indices, lbrow_indices, rbrow_indices, upper_nose_indices,
            lower_nose_indices, leye_indices, reye_indices,
            outer_mouth_indices, inner_mouth_indices)


def line(image, x0, y0, x1, y1, color):
    steep = False
    if x0 < 0 or x0 >= 400 or x1 < 0 or x1 >= 400 or y0 < 0 or y0 >= 400 or y1 < 0 or y1 >= 400:
        return

    if abs(x0 - x1) < abs(y0 - y1):
        x0, y0 = y0, x0
        x1, y1 = y1, x1
        steep = True

    if x0 > x1:
        x0, x1 = x1, x0
        y0, y1 = y1, y0

    for x in range(int(x0), int(x1) + 1):
        t = (x - x0) / float(x1 - x0)
        y = y0 * (1 - t) + y1 * t
        if steep:
            image[x, int(y)] = color
        else:
            image[int(y), x] = color


def draw_landmarks(img, lms):
    try:
        img = img.copy()

        for i, part in enumerate(parts_68[1:]):
            circular = []

            if i in (4, 5, 6, 7):
                circular = [part[0]]

            for p1, p2 in zip(part, list(part[1:]) + circular):
                p1, p2 = lms[p1], lms[p2]

                line(img, p2[1], p2[0], p1[1], p1[0], 1)
    except:
        pass
    return img


def batch_draw_landmarks(imgs, lms):
    return np.array([draw_landmarks(img, l) for img, l in zip(imgs, lms)])


def build_from_caffe(inputs, prototxt_path):
    def prototxt_parser(prototxt_path):

        storage_stack = [[]]
        with open(prototxt_path, 'r') as f:
            for fullline in f.readlines():

                fullline = fullline.strip()
                fullline = fullline.split('#')[0]

                def parse_line(line):

                    if '{' in line:
                        name, rest = line.split('{')
                        storage_stack.append(name.strip().replace('"', ''))
                        storage_stack.append([])

                        line = rest

                    if ':' in line:
                        key, value = line.split(':')
                        rest = ''
                        if '}' in value:
                            value, rest = value.split('}')
                            rest = '}' + rest

                        storage_stack[-1].append((key.strip().replace(
                            '"', ''), value.strip().replace('"', '')))

                        line = rest

                    if '}' in line:
                        data = storage_stack.pop()
                        name = storage_stack.pop()
                        storage_stack[-1].append({
                            name.strip().replace('"', ''): data})

                parse_line(fullline)

        return storage_stack[0]

    def parse_token(token):
        token_dict = {}

        def safe_add_dict(k, v):
            for type_fn in [int, float]:
                try:
                    v = type_fn(v)
                    break
                except:
                    pass

            if k in token_dict:
                if type(token_dict[k]) == list:
                    token_dict[k].append(v)
                else:
                    tv = token_dict[k]
                    token_dict[k] = [tv, v]
            else:
                token_dict[k] = v

        if type(token) == dict:
            for k in token:
                safe_add_dict(k, parse_token(token[k]))

        elif type(token) == list:
            for t in token:
                ptoken = parse_token(t)
                for k in ptoken:
                    safe_add_dict(k, ptoken[k])

        elif type(token) == tuple:
            k, v = token
            safe_add_dict(k, v)

        else:
            return token

        return token_dict

    token = prototxt_parser(prototxt_path)
    net = inputs

    lookup = {
        'data': net
    }

    num_output_pre_scale = 0
    for t in token[5:]:
        node = parse_token(t)['layer']
        if node['type'] == 'BatchNorm':
            net = lookup[node['bottom']]
            net = slim.batch_norm(
                net, center=False, scale=False, scope=node['name'])
        elif node['type'] == 'Convolution':
            num_output = node['convolution_param']['num_output']
            kernel_size = node['convolution_param']['kernel_size']
            pad = node['convolution_param']['pad'] if 'pad' in node['convolution_param'] else 0
            stride = node['convolution_param']['stride'] if 'stride' in node['convolution_param'] else 1
            dilation = node['convolution_param']['dilation'] if 'dilation' in node['convolution_param'] else 1
            num_output_pre_scale = num_output
            net = lookup[node['bottom']]
            net = tf.pad(
                net, [
                    [0, 0],
                    [pad, pad],
                    [pad, pad],
                    [0, 0]
                ])

            net = slim.conv2d(
                net,
                num_output,
                kernel_size,
                stride,
                rate=dilation,
                activation_fn=None,
                padding='VALID',
                scope=node['name']
            )

        elif node['type'] == 'Pooling':
            kernel_size = node['pooling_param']['kernel_size']
            pad = node['pooling_param']['pad']
            stride = node['pooling_param']['stride']

            net = lookup[node['bottom']]
            net = slim.max_pool2d(
                tf.pad(
                    net, [
                         [0, 0],
                        [pad, pad],
                        [pad, pad],
                        [0, 0]
                    ]),
                kernel_size,
                stride, scope=node['name']
            )

        elif node['type'] == 'Power':
            power = node['power_param']['power']
            scale = node['power_param']['scale']
            shift = node['power_param']['shift']

            net = lookup[node['bottom']]
            net = tf.pow(shift + scale * net, power)

        elif node['type'] == 'Scale':
            net = lookup[node['bottom']]
            n_channel = tf.shape(net)[3]
            scale = tf.Variable(
                tf.ones([num_output_pre_scale], tf.float32), name=node['name'] + '/weights')
            net = net * scale

            if node['scale_param']['bias_term'] == 'true':
                net = net + \
                    tf.Variable(
                        tf.zeros([num_output_pre_scale], tf.float32), name=node['name'] + '/biases')

        elif node['type'] == 'Interp':
            zoom_factor = node['interp_param']['zoom_factor']
            pad_beg = node['interp_param']['pad_beg']
            pad_end = node['interp_param']['pad_end']

            net = lookup[node['bottom']]

            net = tf.pad(
                net, [
                    [0, 0],
                    [pad_beg, pad_beg],
                    [pad_beg, pad_beg],
                    [0, 0]
                ])

            in_shape = tf.shape(net)

            net = tf.image.resize_bilinear(net, [
                (in_shape[1] - 1) * zoom_factor + 1,
                (in_shape[2] - 1) * zoom_factor + 1])

            net = tf.pad(
                net, [
                    [0, 0],
                    [pad_end, pad_end],
                    [pad_end, pad_end],
                    [0, 0]
                ])

        elif node['type'] == 'Softmax':
            net = lookup[node['bottom']]
            net = slim.softmax(net)

        elif node['type'] == 'ReLU':
            net = lookup[node['bottom']]
            net = slim.nn.relu(net)

        elif node['type'] == 'Eltwise':
            net1 = lookup[node['bottom'][0]]
            net2 = lookup[node['bottom'][1]]

            op = 1
            if 'eltwise_param' in node:
                if node['eltwise_param']['operation'] == 'PROD':
                    op = 0

            if op == 0:
                net = net1 * net2
            elif op == 1:
                net = net1 + net2
            else:
                raise Exception('Undefined Eltwise Operation')

        elif node['type'] == 'Concat':
            net1 = lookup[node['bottom'][0]]
            net2 = lookup[node['bottom'][1]]

            axis = node['concat_param']['axis']

            net = tf.concat([net1, net2], 3)

        elif node['type'] == 'ArgMax':
            net = lookup[node['bottom']]
            net = tf.arg_max(net, 3)

        elif node['type'] == 'Python':
            net1 = lookup[node['bottom'][0]]
            net2 = lookup[node['bottom'][1]]
            net3 = lookup[node['bottom'][2]]
            net4 = lookup[node['bottom'][3]]

            def combineRegressions(Horizontal, Vertical, HorzRegress, VertRegress):
                HorzReg = np.zeros(Horizontal.shape)
                VertReg = np.zeros(Horizontal.shape)

                Num_Dims = HorzRegress.shape[3]

                for j in range(Num_Dims):
                    if (j > 0):
                        HorzReg = HorzReg + \
                            HorzRegress[:, :, :, j] * (Horizontal == j)
                        VertReg = VertReg + \
                            VertRegress[:, :, :, j] * (Vertical == j)

                HorzReg = ((Horizontal + HorzReg) - 1) / (Num_Dims - 1)
                VertReg = ((Vertical + VertReg) - 1) / (Num_Dims - 1)

                HorzReg[HorzReg < 0] = -1
                VertReg[VertReg < 0] = -1

                return HorzReg.astype(np.float32), VertReg.astype(np.float32)

            net = tf.py_func(combineRegressions, [net1, net2, net3, net4], [
                             tf.float32, tf.float32])
        elif node['type'] == 'Silence':
            node['top'] = 'Silence'
            net = None

        else:
            raise Exception('Undefined behaviour: ' + node['type'])

        if 'functions' in node:
            for fnode in node['functions']:
                if fnode['type'] == 'ReLU':
                    net = slim.nn.relu(net)
                elif fnode['type'] == 'Softmax':
                    net = slim.nn.softmax(net)

        if type(node['top']) != list:
            node['top'] = [node['top']]

        if type(net) != list:
            net = [net]

        for top, n in zip(node['top'], net):
            lookup[top] = n

    return lookup


def build_graph_string(tree, var='net', level=0, strings=['net = inputs']):
    padding = ''.join(['    ' for i in range(level)])

    if tree['name'] == 'nn.Sequential':
        strings.append(padding + 'with tf.name_scope(\'nn.Sequential\'):')
        for tr in tree['children']:
            var = build_graph_string(tr, var, level + 1, strings)
    elif tree['name'] == 'nn.ConcatTable':
        var_table = []
        strings.append(padding + 'with tf.name_scope(\'nn.ConcatTable\'):')

        for idx, tr in enumerate(tree['children']):
            new_var = var + str(idx)
            var_table.append(new_var)
            strings.append(padding + '    {} = {}'.format(new_var, var))
            build_graph_string(tr, new_var, level + 1, strings)
        var = var_table
    elif tree['name'] == 'nn.JoinTable':
        old_var = var[0][:-1]
        strings.append(
            padding + '{} = tf.concat([{}],3)'.format(old_var, ','.join(var)))
        var = old_var
    elif tree['name'] == 'nn.CAddTable':
        old_var = var[0][:-1]
        strings.append(
            padding + '{} = tf.add_n([{}])'.format(old_var, ','.join(var)))
        var = old_var
    elif tree['name'] == 'nn.SpatialConvolution':
        out_channel = int(tree['nOutputPlane'])
        kernal_shape = (int(tree['kH']), int(tree['kW']))
        stride_shape = (int(tree['dH']), int(tree['dW']))

        strings.append(padding + '{0} = tf.pad({0}, np.array([[0,0],[{1:d},{1:d}],[{2:d},{2:d}],[0,0]]))'.format(
            var, int(tree['padH']), int(tree['padW'])
        ))

        strings.append(padding + '{0} = slim.conv2d({0},{1},{2},{3},activation_fn=None,padding=\'VALID\')'.format(
            var, out_channel, kernal_shape, stride_shape
        ))

    elif tree['name'] == 'nn.SpatialFullConvolution':
        out_channel = int(tree['nOutputPlane'])
        kernal_shape = (int(tree['kH']), int(tree['kW']))
        stride_shape = (int(tree['dH']), int(tree['dW']))

        strings.append(padding + '{0} = tf.pad({0}, np.array([[0,0],[{1:d},{1:d}],[{2:d},{2:d}],[0,0]]))'.format(
            var, int(tree['padH']), int(tree['padW'])
        ))

        strings.append(padding + '{0} = slim.conv2d_transpose({0},{1},{2},{3},activation_fn=None,padding=\'VALID\')'.format(
            var, out_channel, kernal_shape, stride_shape
        ))

    elif tree['name'] == 'nn.SpatialBatchNormalization':
        strings.append(padding + '{0} = slim.batch_norm({0})'.format(
            var
        ))
    elif tree['name'] == 'nn.ReLU':
        strings.append(padding + '{0} = slim.nn.relu({0})'.format(
            var
        ))
    elif tree['name'] == 'nn.Sigmoid':
        strings.append(padding + '{0} = slim.nn.sigmoid({0})'.format(
            var
        ))
    elif tree['name'] == 'nn.SpatialMaxPooling':
        strings.append(padding + '{0} = tf.pad({0}, np.array([[0,0],[{1:d},{1:d}],[{2:d},{2:d}],[0,0]]))'.format(
            var, int(tree['padH']), int(tree['padW'])
        ))

        strings.append(padding + '{0} = slim.max_pool2d({0},({1:d},{2:d}),({3:d},{4:d}))'.format(
            var, int(tree['kH']), int(tree['kW']), int(
                tree['dH']), int(tree['dW'])
        ))

    elif tree['name'] == 'nn.Identity':
        pass
    else:
        raise Exception(tree['name'])

    return var


def build_graph(inputs, tree, transpose=(2, 3, 1, 0), layers=[], with_weight=True):

    net = inputs

    if tree['name'] == 'nn.Sequential':
        with tf.name_scope('nn.Sequential'):
            for tr in tree['children']:
                net = build_graph(net, tr, transpose, layers, with_weight)
    elif tree['name'] == 'nn.ConcatTable':
        net_table = []
        with tf.name_scope('nn.ConcatTable'):
            for tr in tree['children']:
                net_table.append(build_graph(
                    net, tr, transpose, layers, with_weight))
        net = net_table
    elif tree['name'] == 'nn.JoinTable':
        net = tf.concat(net, 3)
    elif tree['name'] == 'nn.CAddTable':
        net = tf.add_n(net)
    elif tree['name'] == 'nn.SpatialConvolution':
        out_channel = int(tree['nOutputPlane'])
        kernal_shape = (int(tree['kH']), int(tree['kW']))
        stride_shape = (int(tree['dH']), int(tree['dW']))
        net = tf.pad(
            net, [
                [0, 0],
                [int(tree['padH']), int(tree['padH'])],
                [int(tree['padW']), int(tree['padW'])],
                [0, 0]
            ])
        if 'weight' in tree.keys() and 'bias' in tree.keys() and with_weight:
            net = slim.conv2d(net,
                              out_channel,
                              kernal_shape,
                              stride_shape,
                              activation_fn=None,
                              padding='VALID',
                              weights_initializer=tf.constant_initializer(
                                  tree['weight'].transpose(*transpose)),
                              biases_initializer=tf.constant_initializer(
                                  tree['bias'])
                              )
        else:
            net = slim.conv2d(net,
                              out_channel,
                              kernal_shape,
                              stride_shape,
                              activation_fn=None,
                              padding='VALID'
                              )

        tree['tfname'] = net.name
        tree['tfvar'] = net
    elif tree['name'] == 'nn.SpatialFullConvolution':
        out_channel = int(tree['nOutputPlane'])
        kernal_shape = (int(tree['kH']), int(tree['kW']))
        stride_shape = (int(tree['dH']), int(tree['dW']))
        net = tf.pad(
            net, [
                [0, 0],
                [int(tree['padH']), int(tree['padH'])],
                [int(tree['padW']), int(tree['padW'])],
                [0, 0]
            ])
        if 'weight' in tree.keys() and 'bias' in tree.keys() and with_weight:
            net = slim.conv2d_transpose(net,
                                        out_channel,
                                        kernal_shape,
                                        stride_shape,
                                        activation_fn=None,
                                        padding='VALID',
                                        weights_initializer=tf.constant_initializer(
                                            tree['weight'].transpose(*transpose)),
                                        biases_initializer=tf.constant_initializer(
                                            tree['bias'])
                                        )
        else:
            net = slim.conv2d_transpose(net,
                                        out_channel,
                                        kernal_shape,
                                        stride_shape,
                                        activation_fn=None,
                                        padding='VALID'
                                        )
        tree['tfname'] = net.name
        tree['tfvar'] = net

    elif tree['name'] == 'nn.SpatialBatchNormalization':
        if with_weight:
            net = slim.nn.batch_normalization(net,
                                              tree['running_mean'],
                                              tree['running_var'],
                                              tree['bias'],
                                              tree['weight'],
                                              tree['eps'])
        else:
            net = slim.batch_norm(net)

        tree['tfname'] = net.name
        tree['tfvar'] = net
    elif tree['name'] == 'nn.ReLU':
        net = slim.nn.relu(net)
        tree['tfname'] = net.name
        tree['tfvar'] = net
    elif tree['name'] == 'nn.Sigmoid':
        net = slim.nn.sigmoid(net)
        tree['tfname'] = net.name
        tree['tfvar'] = net
    elif tree['name'] == 'nn.SpatialMaxPooling':
        net = slim.max_pool2d(
            tf.pad(
                net, [
                    [0, 0],
                    [int(tree['padH']), int(tree['padH'])],
                    [int(tree['padW']), int(tree['padW'])],
                    [0, 0]
                ]),
            (int(tree['kH']), int(tree['kW'])),
            (int(tree['dH']), int(tree['dW']))
        )
        tree['tfname'] = net.name
        tree['tfvar'] = net
    elif tree['name'] == 'nn.Identity':
        pass
    else:
        raise Exception(tree['name'])

    return net


def keypts_encoding(keypoints, num_classes):
    keypoints = tf.to_int32(keypoints)
    keypoints = tf.reshape(keypoints, (-1,))
    keypoints = slim.layers.one_hot_encoding(
        keypoints, num_classes=num_classes + 1)
    return keypoints


def get_weight(keypoints, mask=None, ng_w=0.01, ps_w=1.0):
    is_background = tf.equal(keypoints, 0)
    ones = tf.to_float(tf.ones_like(is_background))
    weights = tf.where(is_background, ones * ng_w, ones * ps_w)
    # if mask is not None:
    #     weights *= tf.to_float(mask)

    return weights


def ced_accuracy(t, dists):
    # Head	 Shoulder	Elbow	Wrist	Hip	   Knee	   Ankle
    pts_r = tf.transpose(
        tf.gather(tf.transpose(dists), [8, 12, 11, 10, 2, 1, 0]))
    pts_l = tf.transpose(
        tf.gather(tf.transpose(dists), [9, 13, 14, 15, 3, 4, 5]))
    part_pckh = (tf.to_int32(pts_r <= t) + tf.to_int32(pts_l <= t)) / 2

    return tf.concat([part_pckh, tf.reduce_sum(tf.to_int32(dists <= t), 1)[..., None] / tf.shape(dists)[1]], 1)


def pckh(preds, gts, scales):
    t_range = np.arange(0, 0.51, 0.01)
    dists = tf.sqrt(tf.reduce_sum(tf.pow(preds - gts, 2),
                                  reduction_indices=-1)) / scales
    return ced_accuracy(0.5, dists)


def tf_atan2(y, x):
    angle = tf.where(tf.greater(x, 0.0), tf.atan(y / x), tf.zeros_like(x))
    angle = tf.where(tf.greater(y, 0.0), 0.5 * np.pi - tf.atan(x / y), angle)
    angle = tf.where(tf.less(y, 0.0), -0.5 * np.pi - tf.atan(x / y), angle)
    angle = tf.where(tf.less(x, 0.0), tf.atan(y / x) + np.pi, angle)
    angle = tf.where(tf.logical_and(tf.equal(x, 0.0), tf.equal(y, 0.0)),
                     np.nan * tf.zeros_like(x), angle)

    indices = tf.where(tf.less(angle, 0.0))
    updated_values = tf.gather_nd(angle, indices) + (2 * np.pi)
    update = tf.SparseTensor(indices, updated_values, angle.get_shape())
    update_dense = tf.sparse_tensor_to_dense(update)

    return angle + update_dense


# def import_image(img_path):
#    img = cv2.imread(str(img_path))
#    original_image = Image.init_from_channels_at_back(img[:,:,-1::-1])

#    try:
#        original_image_lms = mio.import_landmark_file('{}/{}.ljson'.format(img_path.parent, img_path.stem)).lms.points.astype(np.float32)
#        original_image.landmarks['LJSON'] = PointCloud(original_image_lms)
#    except:
#        pass

#    return original_image


def crop_image(img, center, scale, res, base=384):
    h = base * scale

    t = Translation(
        [
            res[0] * (-center[0] / h + .5),
            res[1] * (-center[1] / h + .5)
        ]).compose_after(Scale((res[0] / h, res[1] / h))).pseudoinverse()

    # Upper left point
    ul = np.floor(t.apply([0, 0]))
    # Bottom right point
    br = np.ceil(t.apply(res).astype(np.int))

    # crop and rescale

    cimg, trans = img.warp_to_shape(
        br - ul, Translation(-(br - ul) / 2 + (br + ul) / 2), return_transform=True)
    c_scale = np.min(cimg.shape) / np.mean(res)
    new_img = cimg.rescale(1 / c_scale).resize(res)
    return new_img, trans, c_scale


def fit_one(sess, lms_heatmap_prediction, states, img, centre, scale, rotate):

    cimg, trans, c_scale = utils.crop_image(
        img, centre, scale, [384, 384], 400)
    cimg, rtrans = cimg.rotate_ccw_about_centre(rotate, return_transform=True)
    offset = (cimg.shape[0] - 256) // 2
    ccimg, ctrans = cimg.crop(
        (offset, offset), (offset + 256, offset + 256), return_transform=True)

    input_pixels = ccimg.pixels_with_channels_at_back()

    lms_hm_prediction, *pred_states = sess.run(
        [lms_heatmap_prediction] + states,
        feed_dict={'inputs:0': input_pixels[None, ...]})

    bsize, h, w, n_ch = lms_hm_prediction.shape
    lms_hm_prediction_filter = np.stack(list(map(
        lambda x: scipy.ndimage.filters.gaussian_filter(*x),
        zip(lms_hm_prediction.transpose(0, 3, 1, 2).reshape(-1, h, w), [1] * (bsize * n_ch)))))

    hs = np.argmax(np.max(lms_hm_prediction_filter.squeeze(), 2), 1)
    ws = np.argmax(np.max(lms_hm_prediction_filter.squeeze(), 1), 1)
    pts_predictions = np.stack([hs, ws]).T
    original_pred = PointUndirectedGraph(trans.apply(rtrans.apply(
        ctrans.apply(pts_predictions)) * c_scale), utils.bodypose_matrix)

    orig_status = np.array(pred_states).squeeze().transpose(
        0, 3, 1, 2).reshape(-1, 256, 256)
    orig_status = Image(orig_status).warp_to_shape(
        (384, 384), ctrans.pseudoinverse())
    orig_status = orig_status.rescale(c_scale)
    orig_status = orig_status.warp_to_shape(img.shape, trans.pseudoinverse())
    orig_status = orig_status.pixels.reshape(
        4, 7, img.shape[0], img.shape[1]).transpose(0, 2, 3, 1)

    return original_pred, ccimg, PointUndirectedGraph(pts_predictions, utils.bodypose_matrix), lms_hm_prediction_filter, pred_states, orig_status
