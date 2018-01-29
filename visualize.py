from autoencoder.utils import *
from autoencoder.encoder import get_encoder
from autoencoder.decoder import get_decoder
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


parser = argparse.ArgumentParser()
parser.add_argument('--json', default='model_params.json',
                    help='A json defining the mode. Look at example model_params.json '
                     'included for more information. default-model_params.json')
args = parser.parse_args()

def visualize(X, preds):
    assert X.shape[0] == preds.shape[0] and X.shape[0] <=5

    fig = plt.figure(0)
    for i in range(5):
        ax = plt.subplot(2, 5, i+1)
        ax.axis('off')
        if i >= X[i].shape[0]:
            continue
        ax.imshow(X[i], cmap='binary', interpolation=None)
        ax.set_title('input - ' + str(i+1))

    for i in range(5):
        ax = plt.subplot(2, 5, i+6)
        ax.axis('off')
        if i >= preds[i].shape[0]:
            continue
        ax.imshow(preds[i], cmap='binary', interpolation=None)
        ax.set_title('output - ' + str(i+1))

    plt.show()

def test_model(X, model_path, train_params):
    X_ = tf.placeholder(tf.float32, [None]+train_params["img_shape"])

    encoder_out = get_encoder(X_, train_params["encoder_params"])
    decoder_out = get_decoder(encoder_out,
                              train_params["decoder_params"], 
                              train_params["img_shape"])

    loader = tf.train.Saver()

    with tf.Session() as sess:
        loader.restore(sess, model_path)

        preds = sess.run(decoder_out, feed_dict={X_: X})

    visualize(X, preds)

def main(json_file):
    train_params = load_json(json_file)
    imgs = load_data(train_params["data_path"], train_params["img_shape"])

    indices = np.arange(imgs.shape[0])
    np.random.shuffle(indices)

    X = imgs[indices[:5]]

    test_model(X, './model/autoencoder.ckpt', train_params)

if __name__ == '__main__':
    main(args.json)