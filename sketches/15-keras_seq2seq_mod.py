#!/usr/bin/env python3
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
from keras.models import Sequential
from keras.layers import Layer
from keras.layers.core import RepeatVector, Dense, Dropout, Activation
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import EarlyStopping, ModelCheckpoint

import seq2seq
#from seq2seq.models import Seq2seq
from seq2seq.layers.bidirectional import Bidirectional
from seq2seq.layers.encoders import LSTMEncoder
from seq2seq.layers.decoders import LSTMDecoder, LSTMDecoder2, AttentionDecoder

# Network Parameters
# Patents have about 6300 chars/claim
__n_chop = 250
__block_size = 8
__n_hidden = 256

# This limits you to ASCII codes. You can use more if you want, it's just slower
n_input = n_output = 64
char_code_start = -64

# Get claims. Pay attention to the memory layout. The vectors are shared.
def get_book():
    print ("Reading patent claims.")
    with sqlite3.connect("patent-sample.db") as conn:
        claims = conn.execute("SELECT claims FROM patent ORDER BY random();").fetchall()
        book = b''.join([row[0].encode()[:__n_chop].ljust(__n_chop, b'\x00') for row in claims])
        book = np.fromstring(book, dtype=np.uint8).reshape((-1, __n_chop))
        # Only model ascii codes between [char_code_start, char_code_start+n_input]
        book = np.minimum(book - char_code_start, n_input-1)
        # Cut off some samples to be a multiple of block size
        book = book[:(book.shape[0]//__block_size*__block_size), :]
    lens = (book != 0).sum(axis=1, dtype=np.int32) # for seq2seq
    print ("Read {} patents.".format(book.shape[0]))
    return (book, lens)

pat = get_book()

class AttentionSeq2seq(Sequential):

	'''
	This is an attention Seq2seq model based on [3].
	Here, there is a soft allignment between the input and output sequence elements.
	A bidirection encoder is used by default. There is no hidden state transfer in this
	model.
	The  math:
		Encoder:
		X = Input Sequence of length m.
		H = Bidirection_LSTM(X); Note that here the LSTM has return_sequences = True,
		so H is a sequence of vectors of length m.
		Decoder:
        y(i) = LSTM(s(i-1), y(i-1), v(i)); Where s is the hidden state of the LSTM (h and c)
        and v (called the context vector) is a weighted sum over H:
        v(i) =  sigma(j = 0 to m-1)  alpha(i, j) * H(j)
        The weight alpha[i, j] for each hj is computed as follows:
        energy = a(s(i-1), H(j))
        alhpa = softmax(energy)
        Where a is a feed forward network.
	'''
	def __init__(self, output_dim, hidden_dim, output_length, depth=1,bidirectional=True, dropout=0.1, **kwargs):
		if bidirectional and hidden_dim % 2 != 0:
			raise Exception ("hidden_dim for AttentionSeq2seq should be even (Because of bidirectional RNN).")
		super(AttentionSeq2seq, self).__init__()
		if type(depth) not in [list, tuple]:
			depth = (depth, depth)
		if 'batch_input_shape' in kwargs:
			shape = kwargs['batch_input_shape']
			del kwargs['batch_input_shape']
		elif 'input_shape' in kwargs:
			shape = (None,) + tuple(kwargs['input_shape'])
			del kwargs['input_shape']
		elif 'input_dim' in kwargs:
			if 'input_length' in kwargs:
				input_length = kwargs['input_length']
			else:
				input_length = None
			shape = (None, input_length, kwargs['input_dim'])
			del kwargs['input_dim']
		self.add(Layer(batch_input_shape=shape))
		if bidirectional:
			self.add(Bidirectional(LSTMEncoder(output_dim=int(hidden_dim / 2), state_input=False, return_sequences=True, **kwargs)))
		else:
			self.add(LSTMEncoder(output_dim=hidden_dim, state_input=False, return_sequences=True, **kwargs))
		for i in range(0, depth[0] - 1):
			self.add(Dropout(dropout))
			if bidirectional:
				self.add(Bidirectional(LSTMEncoder(output_dim=int(hidden_dim / 2), state_input=False, return_sequences=True, **kwargs)))
			else:
				self.add(LSTMEncoder(output_dim=hidden_dim, state_input=False, return_sequences=True, **kwargs))
		encoder = self.layers[-1]
		self.add(Dropout(dropout))
		self.add(TimeDistributed(Dense(hidden_dim if depth[1] > 1 else output_dim)))
		decoder = AttentionDecoder(hidden_dim=hidden_dim, output_length=output_length, state_input=False, **kwargs)
		self.add(Dropout(dropout))
		self.add(decoder)
		for i in range(0, depth[1] - 1):
			self.add(Dropout(dropout))
			self.add(LSTMEncoder(output_dim=hidden_dim, state_input=False, return_sequences=True, **kwargs))
		self.add(Dropout(dropout))
		self.add(TimeDistributed(Dense(output_dim, activation='softmax')))
		self.encoder = encoder
		self.decoder = decoder

class Seq2seq(Sequential):
	'''
	Seq2seq model based on [1] and [2].
	This model has the ability to transfer the encoder hidden state to the decoder's
	hidden state(specified by the broadcast_state argument). Also, in deep models
	(depth > 1), the hidden state is propogated throughout the LSTM stack(specified by
	the inner_broadcast_state argument. You can switch between [1] based model and [2]
	based model using the peek argument.(peek = True for [2], peek = False for [1]).
	When peek = True, the decoder gets a 'peek' at the context vector at every timestep.
	[1] based model:
		Encoder:
		X = Input sequence
		C = LSTM(X); The context vector
		Decoder:
        y(t) = LSTM(s(t-1), y(t-1)); Where s is the hidden state of the LSTM (h and c)
        y(0) = LSTM(s0, C); C is the context vector from the encoder.
    [2] based model:
		Encoder:
		X = Input sequence
		C = LSTM(X); The context vector
		Decoder:
        y(t) = LSTM(s(t-1), y(t-1), C)
        y(0) = LSTM(s0, C, C)
        Where s is the hidden state of the LSTM (h and c), and C is the context vector
        from the encoder.
	Arguments:
	output_dim : Required output dimension.
	hidden_dim : The dimension of the internal representations of the model.
	output_length : Length of the required output sequence.
	depth : Used to create a deep Seq2seq model. For example, if depth = 3,
			there will be 3 LSTMs on the enoding side and 3 LSTMs on the
			decoding side. You can also specify depth as a tuple. For example,
			if depth = (4, 5), 4 LSTMs will be added to the encoding side and
			5 LSTMs will be added to the decoding side.
	broadcast_state : Specifies whether the hidden state from encoder should be
					  transfered to the deocder.
	inner_broadcast_state : Specifies whether hidden states should be propogated
							throughout the LSTM stack in deep models.
	peek : Specifies if the decoder should be able to peek at the context vector
		   at every timestep.
	dropout : Dropout probability in between layers.
	'''
	def __init__(self, output_dim, hidden_dim, output_length, depth=1, broadcast_state=True, inner_broadcast_state=True, peek=False, dropout=0.1, **kwargs):
		super(Seq2seq, self).__init__()
		if type(depth) not in [list, tuple]:
			depth = (depth, depth)
		if 'batch_input_shape' in kwargs:
			shape = kwargs['batch_input_shape']
			del kwargs['batch_input_shape']
		elif 'input_shape' in kwargs:
			shape = (None,) + tuple(kwargs['input_shape'])
			del kwargs['input_shape']
		elif 'input_dim' in kwargs:
			shape = (None, None, kwargs['input_dim'])
			del kwargs['input_dim']
		lstms = []
		layer = LSTMEncoder(batch_input_shape=shape, output_dim=hidden_dim, state_input=False, return_sequences=depth[0] > 1, **kwargs)
		self.add(layer)
		lstms += [layer]
		for i in range(depth[0] - 1):
			self.add(Dropout(dropout))
			layer = LSTMEncoder(output_dim=hidden_dim, state_input=inner_broadcast_state, return_sequences=i < depth[0] - 2, **kwargs)
			self.add(layer)
			lstms += [layer]
		if inner_broadcast_state:
			for i in range(len(lstms) - 1):
				lstms[i].broadcast_state(lstms[i + 1])
		encoder = self.layers[-1]
		self.add(Dropout(dropout))
		decoder_type = LSTMDecoder2 if peek else LSTMDecoder
		decoder = decoder_type(hidden_dim=hidden_dim, output_length=output_length, state_input=broadcast_state, **kwargs)
		self.add(decoder)
		lstms = [decoder]
		for i in range(depth[1] - 1):
			self.add(Dropout(dropout))
			layer = LSTMEncoder(output_dim=hidden_dim, state_input=inner_broadcast_state, return_sequences=True, **kwargs)
			self.add(layer)
			lstms += [layer]
		if inner_broadcast_state:
				for i in range(len(lstms) - 1):
					lstms[i].broadcast_state(lstms[i + 1])
		if broadcast_state:
			encoder.broadcast_state(decoder)
		self.add(Dropout(dropout))
		self.add(TimeDistributed(Dense(output_dim, activation='softmax')))
		self.encoder = encoder
		self.decoder = decoder

class GiveExamples(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs={}):
        charcodes = self.model.predict(pat[0][:__block_size, :])
        print(charcodes)


model = AttentionSeq2seq(
    batch_input_shape=(__block_size, __n_chop, n_input),
    hidden_dim=__n_hidden, # can be anything
    input_length=__n_chop,
    output_length=__n_chop,
    output_dim=n_output,
    depth=1,
    consume_less="cpu")
#model.add(keras.layers.core.Activation('softmax'))
model = keras.models.Sequential([
    keras.layers.Lambda(
        lambda x: keras.backend.one_hot(keras.backend.cast(x, 'int16'), n_input),
        batch_input_shape=(__block_size, __n_chop,),
        output_shape=(__n_chop, n_input)
    ),
    model
])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = EarlyStopping(monitor='val_loss', patience=2)
checkpoint = ModelCheckpoint("15-weights", save_best_only=True)
give_examples = GiveExamples()
model.fit(
    pat[0],
    np.expand_dims(pat[0], 2),
    batch_size=__block_size,
    callbacks=[early_stopping, checkpoint],
    shuffle=True,
    validation_split=0.1,
)
model.save("15-weights")
