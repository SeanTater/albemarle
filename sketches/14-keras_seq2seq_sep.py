#!/usr/bin/env python3
'''
A Reccurent Neural Network (LSTM) implementation example using TensorFlow library.
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import keras
import numpy as np
import lzma
import re
import sqlite3

# Network Parameters
# Patents have about 6300 chars/claim
__n_chop = 100

# This limits you to ASCII codes. You can use more if you want, it's just slower
n_input = n_output = 128
eye = np.eye(n_input, dtype=np.float32)

# Get claims. Pay attention to the memory layout. The vectors are shared.
def get_book():
    print ("Reading patent claims.")
    with sqlite3.connect("patent-sample.db") as conn:
        claims = conn.execute("SELECT claims FROM patent ORDER BY random();").fetchall()
        book = b''.join([row[0].encode()[:__n_chop].ljust(__n_chop, b'\x00') for row in claims])
        book = np.fromstring(book, dtype=np.uint8).reshape((-1, __n_chop))
        book = np.minimum(book, n_input-1)
    lens = (book != 0).sum(axis=1, dtype=np.int32) # for seq2seq
    print ("Read {} patents.".format(book.shape[0]))
    return (book, lens)

pat = get_book()
pattxt = pat[0]

model = keras.models.Sequential([
    keras.layers.Lambda(
        lambda x: keras.backend.one_hot(keras.backend.cast(x, 'int16'), n_input),
        input_shape=(__n_chop-1,),
        output_shape=(__n_chop-1, n_input)
    ),
    keras.layers.recurrent.GRU(
        input_shape=(__n_chop-1, n_input),
        output_dim=n_output,
        return_sequences=True
    ),
    keras.layers.wrappers.TimeDistributed(keras.layers.core.Dense(
        output_dim=n_output,
        activation='softmax'
    ))
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(
    pat[0][:, :-1],
    np.expand_dims(pat[0][:, 1:], 2),
    shuffle=True,
    validation_split=0.1,
    batch_size=8
)
