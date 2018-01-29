import tensorflow as tf
from autoencoder.utils import DConvConfig


__fc_count = 0
__dconv_count = 0


def dconv(tensor, k_size, n_k, strides, padding, name='d_conv'):
    global __dconv_count
    name = name + str(__dconv_count)
    __dconv_count += 1
    return tf.layers.conv2d_transpose(tensor, n_k, k_size,
                                      strides=strides, padding=padding,
                                      name=name)


def dense(tensor, n_out, name='d_fc'):
    global __fc_count
    name = name + str(__fc_count)
    __fc_count += 1

    m, n_in = tensor.get_shape().as_list()

    W = tf.get_variable(shape=[n_in, n_out], name=name+'_W')
    b = tf.Variable(tf.zeros([1, n_out]), name=name+'_b')

    return tf.add(tf.matmul(tensor, W), b, name=name)


def __add_dconv_layers(tensor, dconv_params):
    dc = DConvConfig(dconv_params)

    for i in range(dc.n_dconvs):
        tensor = dconv(tensor, dc.k_sizes[i], dc.n_ks[i],
                        dc.strides[i], dc.pads[i])
        tensor = tf.nn.relu(tensor)
    return tensor


def __add_fc_layers(tensor, n_outs):

    for i, n_out in enumerate(n_outs):
        tensor = dense(tensor, n_out)
        if i != len(n_outs):
            tensor = tf.nn.relu(tensor)
    return tensor


def __reshape_for_dconv(tensor):
    m, n_in = tensor.get_shape().as_list()
    n_H = n_W = int(n_in ** 0.5)
    return tf.reshape(tensor, shape=[-1, n_H, n_W, 1])


def get_decoder(tensor, decoder_params, img_shape):

    tensor = __add_fc_layers(tensor, decoder_params["fc"])
    tensor = __reshape_for_dconv(tensor)
    tensor = __add_dconv_layers(tensor, decoder_params["dconv"])
    if len(img_shape) == 2:
        tensor = tf.squeeze(tensor, [3])
    return tensor
