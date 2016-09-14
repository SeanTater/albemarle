#!/usr/bin/env python3
'''
A Reccurent Neural Network (LSTM) implementation example using TensorFlow library.
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
from tensorflow.python.client import timeline
import numpy as np
import lzma
import re
import random
import sqlite3
import time

# Parameters
__training_iters = 250000
__batch_size = 128
__layers = 1
display_step = 3
accuracy_step = 10
__cell_kind = lambda t: rnn_cell.MultiRNNCell([rnn_cell.GRUCell(t)] * __layers)
#__cell_kind = rnn_cell.GRUCell

# Network Parameters
#n_input = len(letters) # Unique letters, defined later
#n_output = len(letters)
# Patents have about 6300 chars/claim
__n_chop = 100
__n_steps = __n_chop # timesteps - longest sentence
__n_hidden = 64 # hidden layer num of features

print('\n\t'.join(["- {}:{}".format(k, v)
    for k, v in vars().items()
    if k.startswith("__") and not k.endswith("__")
]))

n_input = n_output = 256
one_hots = tf.constant(np.eye(n_input, dtype=np.float32))



def duplicate(ten):
    return np.hstack([ten, [[0]]*ten.shape[0], ten])

# Get claims. Pay attention to the memory layout. The vectors are shared.
def get_book():
    print ("Reading patent claims.")
    with sqlite3.connect("patent-sample.db") as conn:
        claims = conn.execute("SELECT claims FROM patent ORDER BY random();").fetchall()
        book = b''.join([row[0].encode()[:__n_chop].ljust(__n_chop, b'\x00') for row in claims])
        book = np.fromstring(book, dtype=np.uint8).reshape((-1, __n_chop))
        book = np.minimum(book, 255)
    lens = (book != 0).sum(axis=1, dtype=np.int32) # for seq2seq
    print ("Read {} patents.".format(book.shape[0]))
    return (book, lens)

pat = get_book()
all_indices = list(range(len(pat[0])))
train_indices = all_indices[1000:]
test_indices = all_indices[:1000]
def get_batch(size, test=False):
    indices = random.sample(test_indices if test else train_indices, size)
    pati = [pat[0][i] for i in indices]
    patl = [pat[1][i] for i in indices]
    return (pati, patl)

def RNN(inputs, lens, name, reuse):
    print ("Building network " + name)
    # Define weights
    inputs = tf.gather(one_hots, inputs)
    weights = tf.Variable(tf.random_normal([__n_hidden, n_output]), name=name+"_weights")
    biases = tf.Variable(tf.random_normal([n_output]), name=name+"_biases")

    # Define a lstm cell with tensorflow

    enc_outputs, enc_states = rnn.dynamic_rnn(
        __cell_kind(__n_hidden),
        inputs,
        sequence_length=lens,
        dtype=tf.float32,
        scope=name,
        time_major=False)

    dec_outputs, dec_states = rnn.dynamic_rnn(
        __cell_kind(__n_hidden),
        enc_outputs,
        sequence_length=lens,
        dtype=tf.float32,
        scope=name,
        time_major=False)

    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (__batch_size, __n_steps, n_input)
    # Required shape: '__n_steps' tensors list of shape (__batch_size, n_input)

    '''dec_outputs, dec_states = rnn.rnn(
        __cell_kind(__n_hidden),
        tf.unpack(tf.transpose(inputs, [1, 0, 2])),
        sequence_length=lens,
        dtype=tf.float32,
        scope=name)
    outputs = tf.transpose(tf.pack(outputs), [1, 0, 2])'''
    print ("Done building network " + name)

    # Asserts are actually documentation: they can't be out of date
    assert dec_outputs.get_shape() == (__batch_size, __n_steps, __n_hidden)
    # Linear activation, using rnn output for each char
    # Reshaping here for a `batch` matrix multiply
    # It's faster than `batch_matmul` probably because it can guarantee a
    # static shape
    outputs = tf.reshape(dec_outputs, [__batch_size * __n_steps, __n_hidden])
    finals = tf.matmul(outputs, weights)
    finals = tf.reshape(finals, [__batch_size, __n_steps, n_output]) + biases
    return finals[:, :__n_steps-1, :]

# tf Graph input
pat_chars_i = tf.placeholder(tf.int64, [__batch_size, __n_steps])
pat_lens = tf.placeholder(tf.int32, [__batch_size])

pat_pred = RNN(pat_chars_i, pat_lens, 'pat', None)
pat_chars_i_trim = pat_chars_i[:, 1:]

print (pat_pred.get_shape(), pat_chars_i.get_shape())
assert pat_pred.get_shape() == (__batch_size, __n_chop, n_output)
assert pat_chars_i_trim.get_shape() == (__batch_size, __n_chop)

# Define loss and optimizer
cost = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(pat_pred, pat_chars_i_trim)
    )
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Evaluate model
next_letter_pred = tf.argmax(pat_pred, 2)
correct_pred = tf.equal(next_letter_pred, pat_chars_i_trim)
correct_pred_matters = tf.not_equal(pat_chars_i_trim, 0)
accuracy = (
    tf.reduce_sum(tf.cast(tf.logical_and(correct_pred, correct_pred_matters), tf.float32))
    /
    tf.reduce_sum(tf.cast(correct_pred_matters, tf.float32))
)

# Initializing the variables
init = tf.initialize_all_variables()

def test_it(sess):
    _test_pat_chars_i, _test_pat_lens = get_batch(__batch_size, test=True)
    print ("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={
            pat_chars_i: _test_pat_chars_i,
            pat_lens: _test_pat_lens}))

def train_it(sess, step=1):
    _pat_chars_i, _pat_lens = get_batch(__batch_size)
    inputs = {
        pat_chars_i: _pat_chars_i,
        pat_lens: _pat_lens}
    sess.run(optimizer, feed_dict=inputs)

    if step % display_step == 0:
        # Calculate batch loss
        cost_f = sess.run(cost, feed_dict=inputs)
        print ("{}: Iter {}, cost= {:.6f}".format(
            time.asctime(), str(step*__batch_size), cost_f))


def sample_it(sess):
    ''' This is the craziest way to sample the network.
        It only makes sense to predict letter as you run it.
        But for now we re-run the whole string every time. '''
    pass


# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * __batch_size < __training_iters:
        train_it(sess, step)
        if step % accuracy_step == 0:
            test_it(sess)
        step += 1
    print ("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    test_it(sess)
