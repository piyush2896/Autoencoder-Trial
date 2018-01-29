import tensorflow as tf
from autoencoder.utils import ConvConfig

__conv_count = 0
__fc_count = 0

acts = {
    'softmax': tf.nn.softmax,
    'relu': tf.nn.relu,
    'sigmoid': tf.nn.sigmoid,
    'tanh': tf.nn.tanh
}

def conv2d(tensor, k_size, n_k, strides=2, padding='SAME', name='e_conv'):
    global __conv_count
    name = name + str(__conv_count)
    __conv_count += 1

    m, n_H, n_W, n_C = tensor.get_shape().as_list()

    W = tf.get_variable(shape=[n_H, n_W, n_C, n_k], name=name+'_W')
    b = tf.Variable(tf.zeros([1, 1, 1, n_k]), name=name+'_b')

    return tf.add(tf.nn.conv2d(tensor, W,
                               strides=__to_4dvec(strides),
                               padding=padding), b, name=name)


def dense(tensor, n_out, name='e_fc'):
    global __fc_count
    name = name + str(__fc_count)
    __fc_count += 1

    m, n_in = tensor.get_shape().as_list()

    W = tf.get_variable(shape=[n_in, n_out], name=name+'_W')
    b = tf.Variable(tf.zeros([1, n_out]), name=name+'_b')

    return tf.add(tf.matmul(tensor, W), b, name=name)


def __get_input_node(input_shape):
    assert len(input_shape) == 2 or len(input_shape) == 3

    input_shape = [None] + list(input_shape) 

    X = tf.placeholder(dtype=tf.float32, shape=input_shape, name='input')

    if len(input_shape) == 3:
        X_expanded = tf.expand_dims(X, axis=3)
        return X, X_expanded

    return X, None


def __to_4dvec(strides):
    return [1, strides, strides, 1]


def __add_conv_layers(tensor, conv_params):
    cc = ConvConfig(conv_params)

    for i in range(cc.n_convs):
        tensor = conv2d(tensor, cc.k_sizes[i], cc.n_ks[i],
                        cc.strides[i], cc.pads[i])
        tensor = tf.nn.relu(tensor)
        if cc.apply_pool[i]:
            tensor = tf.nn.max_pool(tensor, ksize=__to_4dvec(cc.pools[i]["k"]), 
                                    strides=__to_4dvec(cc.pools[i]["strides"]),
                                    padding='VALID')
    return tensor


def __add_fc_layers(tensor, n_outs):

    for i, n_out in enumerate(n_outs):
        tensor = dense(tensor, n_out)
        if i != len(n_outs):
            tensor = tf.nn.relu(tensor)
    return tensor


def get_encoder(X, encoder_params):

    X_expanded = None
    if len(X.get_shape().as_list()) == 3:
        X_expanded = tf.expand_dims(X, axis=3)
    #X, X_expanded = __get_input_node(encoder_params["input_shape"])

    if X_expanded is None:
        t = X
    else:
        t = X_expanded

    t = __add_conv_layers(t, encoder_params["conv"])
    t = tf.layers.flatten(t)
    t = __add_fc_layers(t, encoder_params["fc"])
    out = acts[encoder_params["nonlin"]](t, name='e_out')

    return out
