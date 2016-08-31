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

def RNN(tensor, lens, n_hidden, n_summary, name, reuse):
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
        outputs, states = rnn.rnn(lstm_cell, tensor, sequence_length=lens, dtype=tf.float32, scope=scope)
        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']


# Now for parts specific to this data


# Parameters
training_iters = 100000
batch_size = 128
display_step = 10

# Network Parameters
n_input = 300 # GloVe vector size
n_steps = 256 # timesteps - longest sentence
n_hidden = 128 # hidden layer num of features
n_summary = 128

import lzma
import re
import random
def shuffle(x):
    x = x[:]
    random.shuffle(x)
    return x
words = lambda l: re.findall("[a-zA-Z]+|[!?.:;'\"()]", l)


# Get GloVe
veczero = np.zeros(300)
def glove_():
    vecs = np.memmap("glovesmall.arr", np.float32).reshape((-1, 300))
    words = open("glovewords.txt").read().splitlines()
    return dict(zip(words, vecs))
english = glove_()

# Get German word2vec
# veczero = np.zeros(300) -- No deed, same as GloVe!
def germanw2v_():
    vecs = np.memmap("german.vecbin", np.float32).reshape((-1, 300))
    words = open("german.words").read().splitlines()
    return dict(zip(words, vecs))
german = germanw2v_()

# Get verses. Pay attention to the memory layout. The vectors are shared.
def get_book(name, language):
    book = lzma.open('../data/{}-common.vpl.xz'.format(name), 'rt').read().splitlines()
    book = [
        [ language.get(w, veczero) for w in words(l)]
        + ([veczero] * (n_steps - len(words(l))))
        for l in book ]
    lens = np.array([len(l) for l in book], dtype=np.int32)

    for verse in book:
        assert len(verse) <= n_steps, "n_steps should be at least {}".format(len(verse))
    return (book, lens)

kjv = get_book('kjv', english)
kjv_shuf = [ shuffle(x) for x in kjv ]
asv = get_book('asv', english)
elb = get_book('GerElb1905', german)
elb_shuf = [ shuffle(x) for x in elb ]


all_indices = shuffle(list(range(len(kjv[0]))))
train_indices = all_indices[1000:]
test_indices = all_indices[:1000]
print(("all indices", len(all_indices)))
def get_batch(size, test=False):
    indices = random.sample(test_indices if test else train_indices, size)
    kjva = [kjv[0][i] for i in indices]
    kjvl = [kjv[1][i] for i in indices]
    shuf_elbs = [elb_shuf[0][i] for i in indices]
    shuf_elbl = [elb_shuf[1][i] for i in indices]
    elba = [elb[0][i] for i in indices]
    elbl = [elb[1][i] for i in indices]
    return ((kjva, kjvl), (shuf_elbs, shuf_elbl), (elba, elbl))

# tf Graph input
kjv_words = tf.placeholder(tf.float32, [None, None, n_input])
kjv_lens = tf.placeholder(tf.int32, [None])
elb_words = tf.placeholder(tf.float32, [None, None, n_input])
elb_lens = tf.placeholder(tf.int32, [None])
shuf_elb_words = tf.placeholder(tf.float32, [None, None, n_input])
shuf_elb_lens = tf.placeholder(tf.int32, [None])
other_elb_words = tf.placeholder(tf.float32, [None, None, n_input])
other_elb_lens = tf.placeholder(tf.int32, [None])

kjv_pred = RNN(transform_block(kjv_words), kjv_lens, n_hidden, n_summary, 'kjv', None)
elb_pred = RNN(transform_block(elb_words), elb_lens, n_hidden, n_summary, 'elb', None)
shuf_elb_pred = RNN(transform_block(shuf_elb_words), shuf_elb_lens, n_hidden, n_summary, 'elb', True)
cross_elb_pred = RNN(transform_block(other_elb_words), other_elb_lens, n_hidden, n_summary, 'elb', True)

# Define loss and optimizer
#cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
loss_of = lambda a, b: tf.reduce_mean(tf.nn.l2_loss(tf.sub(a, b)))
is_wrong = lambda a: -tf.log(a)
loss_same = loss_of(kjv_pred, elb_pred)       # Same verses have high relation
loss_diff = loss_of(kjv_pred, cross_elb_pred) # Different
loss_shuf = loss_of(kjv_pred, shuf_elb_pred)  # Different
loss_between = loss_of(elb_pred, shuf_elb_pred) # Different order
optimizer = tf.train.AdamOptimizer().minimize(
    tf.log(loss_same)
    + is_wrong(loss_diff)
    + is_wrong(loss_shuf)
    + is_wrong(loss_between)
)

# Evaluate model
#correct_pred = tf.equal(tf.argmax(pred,1), tf.argmax(y,1))
#accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 1
    # Keep training until reach max iterations
    while step * batch_size < training_iters:
        kjv_x, elbs_x, elb_x = get_batch(batch_size)
        kjv_y, elbs_y, elb_y = get_batch(batch_size)
        inputs = {
            kjv_words: kjv_x[0],
            kjv_lens: kjv_x[1],
            elb_words: elb_x[0],
            elb_lens: elb_x[1],
            shuf_elb_words: elbs_x[0],
            shuf_elb_lens: elbs_x[1],
            other_elb_words: elb_y[0],
            other_elb_lens: elb_y[1]
            }

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
