import pandas as pd
import json
import numpy as np
import tensorflow as tf


MAX_DATA_IN_MEMORY = 2000

class ConvConfig(object):

    def __init__(self, conv_params):
        self.k_sizes = conv_params['k_size']
        self.n_ks = conv_params['n_k']
        self.strides = conv_params['strides']
        self.pads = conv_params['padding']
        self.apply_pool = conv_params['do_pool']

        self.pools = []
        for ap in self.apply_pool:
            if ap:
                self.pools.append(conv_params["pool"].pop())
            else:
                self.pools.append(None)

        self.n_convs = len(self.k_sizes)


class DConvConfig(object):

    def __init__(self, dconv_params):
        self.k_sizes = dconv_params['k_size']
        self.n_ks = dconv_params['n_k']
        self.strides = dconv_params['strides']
        self.pads = dconv_params['padding']
        self.n_dconvs = len(self.k_sizes)


def load_data(csv_file, img_shape):
    img_shape = [-1] + list(img_shape)

    df = pd.read_csv(csv_file)

    keys = list(df.keys())
    del keys[0]

    imgs = df[keys].as_matrix()

    imgs = imgs.reshape(img_shape)
    return imgs / 255


def load_json(json_file_path):
    return json.load(open(json_file_path))


def train_dev_split(data, val_split=0.1):
    m = data.shape[0]
    split = int(m * (1-val_split))

    train = data[:split]
    val = data[split:]

    return train, val


def get_batches_generate(data, batch_size=32):
    m = data.shape[0]

    indices = np.arange(m)
    np.random.shuffle(indices)
    data = data[indices]

    batch_dict = {}
    for i in range(0, m, batch_size):
        batch_dict['batch_'+str(i)] = data[i]
    rights = m % batch_size

    indices_right = indices[:-rights].reshape((-1, batch_size))
    data_rights = data[indices_right]
    data_lefts = data[indices[-rights:]]

    for d in data_rights:
        yield d
    yield data_lefts
    #if i < m:
    #    yield data[i:]

    
def get_dataset_tensors(batch_size=32, in_shape=[None, 28, 28]):
    #with tf.device('/cpu'):
    X = tf.placeholder(tf.float32, shape=in_shape)

    dataset = tf.data.Dataset.from_tensor_slices(X)
    dataset = dataset.shuffle(buffer_size=100)
    dataset = dataset.prefetch(100)
    batched_dataset = dataset.batch(batch_size)

    return X, batched_dataset


def get_iterator(dataset):
    return dataset.make_initializable_iterator()


def get_data_in_chunks(data):
    m = data.shape[0]

    indices = np.arange(m)
    np.random.shuffle(indices)
    data = data[indices]

    if m < MAX_DATA_IN_MEMORY:
        return data

    for i in range(0, m, MAX_DATA_IN_MEMORY):
        yield data[i:i+MAX_DATA_IN_MEMORY]

    if i < m:
        yield data[i:]
