import numpy as np
import tensorflow as tf
import menpo.io as mio
from menpo.image import Image
from menpo.shape import PointCloud

from menpo.transform import Translation, Scale
from menpo.shape import PointCloud
from scipy.sparse import csr_matrix

slim = tf.contrib.slim

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
