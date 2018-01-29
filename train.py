import numpy as np
import tensorflow as tf
from autoencoder.utils import *
from autoencoder.encoder import get_encoder
from autoencoder.decoder import get_decoder
from tqdm import tqdm
import shutil
import argparse
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


parser = argparse.ArgumentParser()
parser.add_argument('--json', default='model_params.json',
                    help='A json defining the mode. Look at example model_params.json '
                     'included for more information. default-model_params.json')
args = parser.parse_args()


def main():
    print("Loading Data and Params...")
    train_params = load_json(args.json)
    imgs = load_data(train_params["data_path"], train_params["img_shape"])

    learning_rate = train_params["lr"]
    batch_size = train_params["batch_size"]
    val_split = train_params["split"]
    epochs = train_params["epochs"]

    print("Spliting Data...")
    train_data, dev_data = train_dev_split(imgs, val_split)
    train_size = train_data.shape[0]
    dev_size = dev_data.shape[0]
    n_train_chunks = train_size // MAX_DATA_IN_MEMORY + (1 if train_size % MAX_DATA_IN_MEMORY != 0 else 0)

    print("Loading to Iterator...")
    X, dataset = get_dataset_tensors(batch_size,
                                     in_shape=[None]+train_params["img_shape"])
    iterator = get_iterator(dataset)
    next_element = iterator.get_next()

    iterator_init_op = iterator.initializer

    print("Compiling Architecture...")
    encoder_out = get_encoder(next_element, train_params["encoder_params"])
    decoder_out = get_decoder(encoder_out,
                              train_params["decoder_params"], 
                              train_params["img_shape"])

    loss = tf.losses.mean_squared_error(next_element, decoder_out)
    train_step = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

    print("Start Training...")
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            print('Epoch {}/{}'.format(epoch+1, epochs))
            train_data_chunks = get_data_in_chunks(train_data)
            dev_data_chunks = get_data_in_chunks(dev_data)

            loss_ = []
            dev_loss = []

            for X_train in tqdm(train_data_chunks, total=n_train_chunks):
                sess.run(iterator_init_op, feed_dict={X: X_train})
                while True:
                    try:
                        _, l = sess.run([train_step, loss])
                        loss_.append(l)
                    except tf.errors.OutOfRangeError:
                        break
            
            for X_test in dev_data_chunks:
                sess.run(iterator_init_op, feed_dict={X:X_test})
                while True:
                    try:
                        dev_loss.append(sess.run(loss))
                    except tf.errors.OutOfRangeError:
                        break

            print('Loss: {}\t Val Loss: {}'.format(np.mean(loss_), np.mean(dev_loss)))
            cols, _ = tuple(shutil.get_terminal_size((80, 20)))
            print('-'*cols)
        saver.save(sess, './model/autoencoder.ckpt')
        print('\n\nModel Saved!!')

if __name__ == '__main__':
    main()
