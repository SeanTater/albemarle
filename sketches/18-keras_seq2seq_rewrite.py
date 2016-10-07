#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
A Reccurent Neural Network (LSTM) implementation example using TensorFlow library.
Long Short Term Memory paper: http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

import numpy as np
import lzma
import re
import sqlite3

import keras
import os
from keras.models import Sequential, Model
from keras.layers import Input, Layer
from keras.layers.core import RepeatVector, Dense, Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.recurrent import LSTM, GRU
from keras import backend as K
from keras.engine import InputSpec

import seq2seq
#from seq2seq.models import Seq2seq
from seq2seq.layers.encoders import LSTMEncoder
from seq2seq.layers.decoders import LSTMDecoder, LSTMDecoder2, AttentionDecoder
from seq2seq.layers.bidirectional import Bidirectional

# Network Parameters
# Patents have about 6300 chars/claim
__n_chop = 50
bs = __block_size = 256
__n_hidden = 512
__pat_cap = 250000
__slice_cap = 10000000

# This limits you to ASCII codes. You can use more if you want, it's just slower
n_input = n_output = 128
char_code_start = 0

# Get claims. Pay attention to the memory layout. The vectors are shared.
def get_book():
    print ("Reading patent claims.")
    with sqlite3.connect(os.path.expanduser("~/patent-sample.db")) as conn:
        claims = conn.execute(
            "SELECT claims FROM patent where length(claims) >= 50 ORDER BY rand LIMIT ?;",
            [__pat_cap]).fetchall()
        slicelen = __n_chop * __block_size
        evenout = lambda x: x[:len(x)//slicelen*slicelen]
        book = ''.join([row[0] for row in claims]).encode()
        book = np.fromstring(evenout(book), dtype=np.uint8).reshape((-1, __n_chop))[:__slice_cap, :]
        # Only model ascii codes between [char_code_start, char_code_start+n_input]
        book = np.minimum(book - char_code_start, n_input-1)
    print ("Read {} patent slices.".format(book.shape[0]))
    return book
pat = get_book()

class GiveExamples(keras.callbacks.Callback):
    def __init__(self):
        self.slice_start=0
    
    def on_epoch_end(self, batch, logs={}):
        snippet = lambda s: s.tostring()[:50].decode(errors='ignore').replace('\n', '<NL>')
        sample = pat[self.slice_start:self.slice_start+3, :-1]
        self.slice_start += 3
        
        chars = model.predict(
            sample,
            batch_size=bs
            ).argmax(axis=2).astype(np.uint8)
        print('\n' + 
            '\n'.join(
                [ snippet(st) for st in chars]
                + [snippet(st) for st in sample]))

model = Sequential([
    keras.layers.Lambda(
        lambda x: keras.backend.one_hot(keras.backend.cast(x, 'int8'), n_input),
        input_shape=(__n_chop-1,),
        output_shape=(__n_chop-1, n_input)
    ),
    LSTM(__n_hidden, input_shape=(__n_chop-1, n_input), return_sequences=True, dropout_W=0.25, dropout_U=0.25),
    BatchNormalization(),
    LSTM(__n_hidden, input_shape=(__n_chop-1, n_input), return_sequences=True, dropout_W=0.25, dropout_U=0.25),
    BatchNormalization(),
    #keras.layers.core.RepeatVector(__n_chop-1),
    LSTM(n_output, return_sequences=True)
])
model.add(TimeDistributed(Activation('softmax')))
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=[keras.metrics.sparse_categorical_accuracy]
)



model.fit(
    pat[:, :-1],
    #np.eye(128, dtype=np.float32)[pat[:, :-1]],
    #np.eye(128, dtype=np.float32)[pat[:,  1:]],
    np.expand_dims(pat[:, 1:], 2),
    batch_size=__block_size,
    callbacks=[
        ModelCheckpoint("15-weights", save_best_only=True),
        #EarlyStopping(monitor='val_loss', patience=2),
        GiveExamples()
    ],
    shuffle=True,
    validation_split=0.01,
    nb_epoch=10
)
#model.save("15-weights")
