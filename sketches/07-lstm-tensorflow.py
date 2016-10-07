#!/usr/bin/env python3
'''
A Reccurent Neural Network (LSTM) implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits (http://yann.lecun.com/exdb/mnist/)
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np

def transform_block(tensor):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (batch_size, n_steps, n_input)
    # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

    # Permuting batch_size and n_steps
    tensor = tf.transpose(tensor, [1, 0, 2])
    # Reshaping to (n_steps*batch_size, n_input)
    tensor = tf.reshape(tensor, [-1, n_input])
    # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
    return tf.split(0, n_steps, tensor)

def RNN(tensor, n_hidden, n_summary, name, reuse):
    with tf.variable_scope(name, reuse) as scope:
        # Define weights
        weights = {
            'out': tf.Variable(tf.random_normal([n_hidden, n_summary]), name=name+"_weights")
        }
        biases = {
            'out': tf.Variable(tf.random_normal([n_summary]), name=name+"_biases")
        }

        # Define a lstm cell with tensorflow
        lstm_cell = rnn_cell.LSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
        # Get lstm cell output
        outputs, states = rnn.rnn(lstm_cell, tensor, dtype=tf.float32, scope=scope)
        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']


# Now for parts specific to this data


# Parameters
training_iters = 100000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 102 # possible symbols
n_steps = 1300 # timesteps - longest sentence
n_hidden = 128 # hidden layer num of features
n_summary = 128

import lzma
import re
import random
def shuffle(x):
    x = x[:]
    random.shuffle(x)
    return x

letters_out = " abcdefghijklmnopqrstuvwzyxABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    + "~`!@#$%^&*()_+-={}|[]\:\";'<>?,./ßäöüÄÖÜ"
letters_in = letters_out[1:]
one_hots = map(lambda i: tf.one_hot())
english = german = lambda l: letters.find(l) + 1 # unknown = space = first index


# Get verses. Pay attention to the memory layout. The vectors are shared.
def get_book(name, language):
    book = lzma.open('../data/{}-common.vpl.xz'.format(name), 'rt').read().splitlines()
    book = [
        [ language(w, 0) for w in l]
        + ([veczero] * (n_steps - len(l)))
        for l in book ]
    for verse in book:
        assert len(verse) <= n_steps, "n_steps should be at least {}".format(len(verse))
    return book

kjv = get_book('kjv', english)
kjv_shuf = [ shuffle(x) for x in kjv ]
asv = get_book('asv', english)
elb = get_book('GerElb1905', german)
elb_shuf = [ shuffle(x) for x in elb ]


all_indices = shuffle(list(range(len(kjv))))
train_indices = all_indices[1000:]
test_indices = all_indices[:1000]
def get_batch(size, test=False):
    indices = random.sample(test_indices if test else train_indices, size)
    kjva = np.array([kjv[i] for i in indices])
    elbs = np.array([elb_shuf[i] for i in indices])
    elba = np.array([elb[i] for i in indices])
    return (kjva, elbs, elba)

# tf Graph input
kjv_words = tf.placeholder(tf.float32, [None, n_steps, n_input])
elb_words = tf.placeholder(tf.float32, [None, n_steps, n_input])
shuf_elb_words = tf.placeholder(tf.float32, [None, n_steps, n_input])
other_elb_words = tf.placeholder(tf.float32, [None, n_steps, n_input])

kjv_pred = RNN(transform_block(kjv_words), n_hidden, n_summary, 'kjv', None)
elb_pred = RNN(transform_block(elb_words), n_hidden, n_summary, 'elb', None)
shuf_elb_pred = RNN(transform_block(shuf_elb_words), n_hidden, n_summary, 'elb', True)
cross_elb_pred = RNN(transform_block(other_elb_words), n_hidden, n_summary, 'elb', True)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))


# Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        kjv_x, elb_s, elb_x = get_batch(batch_size)
        kjv_y, elb_s, elb_y = get_batch(batch_size)
        inputs = {
            kjv_words: kjv_x, elb_words: elb_x,
            shuf_elb_words: elb_s, other_elb_words: elb_y}

        #batch_x, batch_y = mnist.train.next_batch(batch_size)
        # Reshape data to get 28 seq of 28 elements
        #batch_x = batch_x.reshape((batch_size, n_steps, n_input))
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict=inputs)
        if step % display_step == 0:
            # Calculate batch loss
            loss_same_f = sess.run(loss_same, feed_dict=inputs)
            loss_diff_f = sess.run(loss_diff, feed_dict=inputs)
            loss_shuf_f = sess.run(loss_shuf, feed_dict=inputs)
            loss_between_f = sess.run(loss_between, feed_dict=inputs)
            print ("Iter {}, Loss= {:.6f}, Loss diff ={:.6f}, Loss shuf = {:.6f}, loss between = {:.6f}".format(
                str(step*batch_size), loss_same_f, loss_diff_f, loss_shuf_f, loss_between_f))
        step += 1
    print ("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    #test_len = 128
    #test_data = mnist.test.images[:test_len].reshape((-1, n_steps, n_input))
    #test_label = mnist.test.labels[:test_len]
    #print ("Testing Accuracy:", \
    #    sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
