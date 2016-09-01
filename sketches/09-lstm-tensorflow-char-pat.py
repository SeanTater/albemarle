#!/usr/bin/env python3
'''
A Reccurent Neural Network (LSTM) implementation example using TensorFlow library.
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import numpy as np
import lzma
import re
import random
import sqlite3

# Parameters
__training_iters = 250000
__batch_size = 128
display_step = 10
accuracy_step = 100
__cell_kind = rnn_cell.GRUCell

# Network Parameters
#n_input = len(letters) # Unique letters, defined later
#n_output = len(letters)
# Patents have about 6300 chars/claim
__n_steps = 200 # timesteps - longest sentence
__n_hidden = 256 # hidden layer num of features

print('\n\t'.join(
    ["Two layer GRU"]
    + ["- {}:{}".format(k, v)
    for k, v in vars().items()
    if k.startswith("__") and not k.endswith("__")
]))

def shuffle(x):
    x = x[:]
    random.shuffle(x)
    return x


letters = (" abcdefghijklmnopqrstuvwzyxABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    + "~`!@#$%^&*()_+-={}|[]\:\";'<>?,./ßäöüÄÖÜ")
n_input = n_output = len(letters)
_letters_in = letters[1:] # Only use me if you know what you are doing!
one_hots = np.eye(len(letters))
english = lambda l: _letters_in.find(l) + 1 # unknown = space = first index


# Get claims. Pay attention to the memory layout. The vectors are shared.
def get_book(language):
    print ("Reading patent claims.")
    with sqlite3.connect("patent-sample.db") as conn:
        book = [row[0] for row in conn.execute("SELECT claims FROM patent ORDER BY random();").fetchall()]

    book = [txt[:__n_steps].ljust(__n_steps) for txt in book]
    booki = np.array([
        [ language(w) for w in l]
        for l in book ])
    book = one_hots[booki]
    lens = np.array([len(l) for l in book], dtype=np.int32)

    print ("Done reading.")
    return (book, booki, lens)

pat = get_book(english)

all_indices = shuffle(list(range(len(pat[0]))))
train_indices = all_indices[1000:]
test_indices = all_indices[:1000]
print(("all indices", len(all_indices)))
def get_batch(size, test=False):
    indices = random.sample(test_indices if test else train_indices, size)
    pata = [pat[0][i] for i in indices]
    pati = [pat[1][i] for i in indices]
    patl = [pat[2][i] for i in indices]
    return (pata, pati, patl)



def transform_block(tensor):
    # Prepare data shape to match `rnn` function requirements
    # Current data input shape: (__batch_size, __n_steps, n_input)
    # Required shape: '__n_steps' tensors list of shape (__batch_size, n_input)

    # Permuting __batch_size and __n_steps
    tensor = tf.transpose(tensor, [1, 0, 2])
    # Reshaping to (__n_steps*__batch_size, n_input)
    tensor = tf.reshape(tensor, [-1, n_input])
    # Split to get a list of '__n_steps' tensors of shape (__batch_size, n_input)
    return tf.split(0, __n_steps, tensor)

def split_last_axis(tensor):
    # Reshaping to (__n_steps*__batch_size, n_input)
    tensor = tf.reshape(tensor, [-1, n_input])
    # Split to get a list of '__n_steps' tensors of shape (__batch_size, n_input)
    return tf.split(0, __n_steps, tensor)

def RNN(tensor, lens, name, reuse):
    print ("Building network " + name)
    with tf.variable_scope(name, reuse) as scope:
        # Define weights
        weights = {
            'out': tf.Variable(tf.random_normal([__n_hidden, n_output]), name=name+"_weights")
        }
        biases = {
            'out': tf.Variable(tf.random_normal([n_output]), name=name+"_biases")
        }

        # Define a lstm cell with tensorflow
        cell = __cell_kind(__n_hidden)
        # Get lstm cell output
        outputs, states = rnn.rnn(cell, tensor, sequence_length=lens, dtype=tf.float32, scope=scope)
        print ("Done building network " + name)
        # Linear activation, using rnn inner loop output for each char
        return tf.batch_matmul(outputs, tf.tile(tf.expand_dims(weights['out'], 0), [__n_steps, 1, 1])) + biases['out']

# tf Graph input
pat_chars = tf.placeholder(tf.float32, [None, __n_steps, n_input])
pat_chars_i = tf.placeholder(tf.int64, [None, __n_steps])
pat_chars_t = tf.transpose(pat_chars_i, [1,0])
pat_lens = tf.placeholder(tf.int32, [None])

pat_pred_0 = RNN(transform_block(pat_chars), pat_lens, 'pat0', None)
pat_pred = RNN(split_last_axis(pat_pred_0), pat_lens, 'pat', None)

print (pat_pred.get_shape(), pat_chars_t.get_shape())
# Define loss and optimizer
cost = tf.reduce_mean(
    tf.nn.sparse_softmax_cross_entropy_with_logits(pat_pred[:__n_steps-1, :, :], pat_chars_t[1:, :])
    )
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Evaluate model
next_letter_pred = tf.argmax(pat_pred[:__n_steps-1, :, :], 2)
correct_pred = tf.equal(next_letter_pred, pat_chars_t[1:, :])
correct_pred_matters = tf.not_equal(pat_chars_t[1:, :], 0)
accuracy = (
    tf.reduce_sum(tf.cast(tf.logical_and(correct_pred, correct_pred_matters), tf.float32))
    /
    tf.reduce_sum(tf.cast(correct_pred_matters, tf.float32))
)

# Initializing the variables
init = tf.initialize_all_variables()

def test_it(sess):
    _test_pat_chars, _test_pat_chars_i, _test_pat_lens = get_batch(__batch_size, test=True)
    print ("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={
            pat_chars: _test_pat_chars,
            pat_chars_i: _test_pat_chars_i,
            pat_lens: _test_pat_lens}))

def train_it(sess, step=1):
    _pat_chars, _pat_chars_i, _pat_lens = get_batch(__batch_size)
    inputs = {
        pat_chars: _pat_chars,
        pat_chars_i: _pat_chars_i,
        pat_lens: _pat_lens}

    # Run optimization op (backprop)
    sess.run(optimizer, feed_dict=inputs)

    if step % display_step == 0:
        # Calculate batch loss
        cost_f = sess.run(cost, feed_dict=inputs)
        print ("Iter {}, cost= {:.6f}".format(
            str(step*__batch_size), cost_f))

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
