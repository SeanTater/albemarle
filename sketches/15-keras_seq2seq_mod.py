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
from keras.models import Sequential
from keras.layers import Layer
from keras.layers.core import RepeatVector, Dense, Dropout, Activation
from keras.layers.wrappers import TimeDistributed
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.recurrent import time_distributed_dense

import seq2seq
#from seq2seq.models import Seq2seq
from seq2seq.layers.bidirectional import Bidirectional
from seq2seq.layers.encoders import LSTMEncoder
from seq2seq.layers.decoders import AttentionDecoder

# Network Parameters
# Patents have about 6300 chars/claim
__n_chop = 250
__block_size = 8
__n_hidden = 128

# This limits you to ASCII codes. You can use more if you want, it's just slower
n_input = n_output = 128
char_code_start = 0

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

#from __future__ import absolute_import
from keras import backend as K
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras import activations, initializations
from keras.engine import InputSpec
#import numpy as np


def get_state_transfer_rnn(RNN):
    '''Converts a given Recurrent sub class (e.g, LSTM, GRU) to its state transferable version.
    A state transfer RNN can transfer its hidden state to another one of the same type and compatible dimensions.
    '''

    class StateTransferRNN(RNN):

        def __init__(self, state_input=True, **kwargs):
            self.state_outputs = []
            self.state_input = state_input
            super(StateTransferRNN, self).__init__(**kwargs)

        def reset_states(self):
            stateful = self.stateful
            self.stateful = stateful or self.state_input or len(self.state_outputs) > 0
            if self.stateful:
                super(StateTransferRNN, self).reset_states()
            self.stateful = stateful

        def build(self,input_shape):
            stateful = self.stateful
            self.stateful = stateful or self.state_input or len(self.state_outputs) > 0
            super(StateTransferRNN, self).build(input_shape)
            self.stateful = stateful

        def broadcast_state(self, rnns):
            rnns = (set if type(rnns) in [list, tuple] else lambda a: {a})(rnns)
            rnns -= set(self.state_outputs)
            self.state_outputs.extend(rnns)
            for rnn in rnns:
                rnn.state_input = self
                rnn.updates = getattr(rnn, 'updates', [])
                rnn.updates.extend(zip(rnn.states, self.states_to_transfer))

        def call(self, x, mask=None):
            last_output, outputs, states = K.rnn(
                self.step,
                self.preprocess_input(x),
                self.states or self.get_initial_states(x),
                go_backwards=self.go_backwards,
                mask=mask,
                constants=self.get_constants(x),
                unroll=self.unroll,
                input_length=self.input_spec[0].shape[1])
            self.updates = zip(self.states, states)
            self.states_to_transfer = states
            return outputs if self.return_sequences else last_output
    return StateTransferRNN


StateTransferSimpleRNN = get_state_transfer_rnn(SimpleRNN)
StateTransferGRU = get_state_transfer_rnn(GRU)
StateTransferLSTM = get_state_transfer_rnn(LSTM)

class LSTMEncoder(StateTransferLSTM):

	def __init__(self, decoder=None, decoders=[], **kwargs):
		super(LSTMEncoder, self).__init__(**kwargs)
		if decoder:
			decoders = [decoder]
		self.broadcast_state(decoders)


class LSTMDecoder(StateTransferLSTM):
    '''
    A basic LSTM decoder. Similar to [1].
    The output of at each timestep is the input to the next timestep.
    The input to the first timestep is the context vector from the encoder.
    Basic equation:
        y(t) = LSTM(s(t-1), y(t-1)); Where s is the hidden state of the LSTM (h and c)
        y(0) = LSTM(s0, C); C is the context vector from the encoder.
    In addition, the hidden state of the encoder is usually used to initialize the hidden
    state of the decoder. Checkout models.py to see how its done.
    '''
    input_ndim = 2

    def __init__(self, output_length, hidden_dim=None, **kwargs):
        self.output_length = output_length
        self.hidden_dim = hidden_dim
        input_dim = None
        if 'input_dim' in kwargs:
            kwargs['output_dim'] = input_dim
        if 'input_shape' in kwargs:
            kwargs['output_dim'] = kwargs['input_shape'][-1]
        elif 'batch_input_shape' in kwargs:
            kwargs['output_dim'] = kwargs['batch_input_shape'][-1]
        elif 'output_dim' not in kwargs:
            kwargs['output_dim'] = None
        super(LSTMDecoder, self).__init__(**kwargs)
        self.return_sequences = True
        self.updates = []
        self.consume_less = 'mem'

    def build(self, input_shape):
        input_shape = list(input_shape)
        input_shape = input_shape[:1] + [self.output_length] + input_shape[1:]
        if not self.hidden_dim:
            self.hidden_dim = input_shape[-1]
        output_dim = input_shape[-1]
        self.output_dim = self.hidden_dim
        initial_weights = self.initial_weights
        self.initial_weights = None
        super(LSTMDecoder, self).build(input_shape)
        self.output_dim = output_dim
        self.initial_weights = initial_weights
        self.W_y = self.init((self.hidden_dim, self.output_dim), name='{}_W_y'.format(self.name))
        self.b_y = K.zeros((self.output_dim), name='{}_b_y'.format(self.name))
        self.trainable_weights += [self.W_y, self.b_y]
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        input_shape.pop(1)
        self.input_spec = [InputSpec(shape=tuple(input_shape))]

    def get_constants(self, x):
        output_dim = self.output_dim
        self.output_dim = self.hidden_dim
        consts = super(LSTMDecoder, self).get_constants(x)
        self.output_dim = output_dim
        return consts

    def reset_states(self):
        output_dim = self.output_dim
        self.output_dim = self.hidden_dim
        super(LSTMDecoder, self).reset_states()
        self.output_dim = output_dim

    def get_initial_states(self, x):
        output_dim = self.output_dim
        self.output_dim = self.hidden_dim
        initial_states = super(LSTMDecoder, self).get_initial_states(x)
        self.output_dim = output_dim
        return initial_states

    def step(self, x, states):
        assert len(states) == 5, len(states)
        states = list(states)
        y_tm1 = states.pop(2)
        output_dim = self.output_dim
        self.output_dim = self.hidden_dim
        h_t, new_states = super(LSTMDecoder, self).step(y_tm1, states)
        self.output_dim = output_dim
        y_t = self.activation(K.dot(h_t, self.W_y) + self.b_y)
        new_states += [y_t]
        return y_t, new_states

    def call(self, x, mask=None):
        X = K.repeat(x, self.output_length)
        input_shape = list(self.input_spec[0].shape)
        input_shape = input_shape[:1] + [self.output_length] + input_shape[1:]
        self.input_spec = [InputSpec(shape=tuple(input_shape))]
        if self.stateful or self.state_input or len(self.state_outputs) > 0:
            initial_states = self.states[:]
        else:
            initial_states = self.get_initial_states(X)
        constants = self.get_constants(X)
        y_0 = K.permute_dimensions(X, (1, 0, 2))[0, :, :]
        initial_states += [y_0]
        last_output, outputs, states = K.rnn(self.step, X,
                                             initial_states,
                                             go_backwards=self.go_backwards,
                                             mask=mask,
                                             constants=constants,
                                             unroll=self.unroll,
                                             input_length=self.output_length)
        if self.stateful and not self.state_input:
            self.updates = []
            for i in range(2):
                self.updates.append((self.states[i], states[i]))
        self.states_to_transfer = states
        input_shape.pop(1)
        self.input_spec = [InputSpec(shape=input_shape)]
        return outputs

    def assert_input_compatibility(self, x):
        shape = x._keras_shape
        assert K.ndim(x) == 2, "LSTMDecoder requires 2D  input, not " + str(K.ndim(x)) + "D."
        assert shape[-1] == self.output_dim or not self.output_dim, "output_dim of LSTMDecoder should be same as the last dimension in the input shape. output_dim = "+ str(self.output_dim) + ", got tensor with shape : " + str(shape) + "."

    def get_output_shape_for(self, input_shape):
        input_shape = list(input_shape)
        output_shape = input_shape[:1] + [self.output_length] + input_shape[1:]
        return tuple(output_shape)

    def get_config(self):
        config = {'name': self.__class__.__name__,
        'hidden_dim': self.hidden_dim,
        'output_length': self.output_length}
        base_config = super(LSTMDecoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

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


class LSTMDecoder2(LSTMDecoder):
    '''
    This decoder is similar to the first one, except that at every timestep the decoder gets
    a peek at the context vector.
    Similar to [2].
    Basic equation:
        y(t) = LSTM(s(t-1), y(t-1), C)
        y(0) = LSTM(s0, C, C)
        Where s is the hidden state of the LSTM (h and c), and C is the context vector
        from the encoder.
    '''
    def build(self, input_shape):
        initial_weights = self.initial_weights
        self.initial_weights = None
        super(LSTMDecoder2, self).build(input_shape)
        self.initial_weights = initial_weights
        dim = self.input_spec[0].shape[-1]
        self.W_x = self.init((dim, dim), name='{}_W_x'.format(self.name))
        self.b_x = K.zeros((dim,), name='{}_b_x'.format(self.name))
        self.trainable_weights += [self.W_x, self.b_x]
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def step(self, x, states):
        assert len(states) == 5, len(states)
        states = list(states)
        y_tm1 = states.pop(2)
        v = self.activation(K.dot(x, self.W_x) + self.b_x)
        y_tm1 += v
        output_dim = self.output_dim
        self.output_dim = self.hidden_dim
        h_t, new_states = super(LSTMDecoder, self).step(y_tm1, states)
        self.output_dim = output_dim
        y_t = self.activation(K.dot(h_t, self.W_y) + self.b_y)
        new_states += [y_t]
        return y_t, new_states

    def get_config(self):
        config = {'name': self.__class__.__name__}
        base_config = super(LSTMDecoder2, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class AttentionDecoder(LSTMDecoder2):
    '''
    This is an attention decoder based on [3].
    Unlike the other decoders, AttentionDecoder requires the encoder to return
    a sequence of hidden states, instead of just the final context vector.
    Or in Keras language, while using AttentionDecoder, the encoder should have
    return_sequences = True.
    Also, the encoder should be a bidirectional RNN for best results.
    Working:
    A sequence of vectors X = {x0, x1, x2,....xm-1}, where m = input_length is input
    to the encoder.
    The encoder outputs a hidden state at each timestep H = {h0, h1, h2,....hm-1}
    The decoder uses H to generate a sequence of vectors Y = {y0, y1, y2,....yn-1},
    where n = output_length
    Decoder equations:
        Note: hk means H(k).
        y(i) = LSTM(s(i-1), y(i-1), v(i)); Where s is the hidden state of the LSTM (h and c)
        and v (called the context vector) is a weighted sum over H:
        v(i) =  sigma(j = 0 to m-1)  alpha[i, j] * hj
        The weight alpha(i, j) for each hj is computed as follows:
        energy = a(s(i-1), hj)
        alhpa = softmax(energy)
        Where a is a feed forward network.
    '''

    input_ndim = 3

    def build(self, input_shape):
        self.input_length = input_shape[1]
        if not self.input_length:
            raise Exception ('AttentionDecoder requires input_length.')
        initial_weights = self.initial_weights
        self.initial_weights = None
        super(AttentionDecoder, self).build(input_shape[:1] + input_shape[2:])
        self.initial_weights = initial_weights
        dim = self.input_dim
        hdim = self.hidden_dim
        self.W_h = self.init((hdim, dim), name='{}_W_h'.format(self.name))
        self.b_h = K.zeros((dim, ), name='{}_b_h'.format(self.name))
        self.W_a = self.init((dim, 1), name='{}_W_a'.format(self.name))
        self.b_a = K.zeros((1,), name='{}_b_a'.format(self.name))
        self.trainable_weights += [self.W_a, self.b_a, self.W_h, self.b_h]
        if self.initial_weights is not None:
            self.set_weights(self.inital_weights)
            del self.initial_weights

    def step(self, x, states):
        h_tm1, c_tm1, y_tm1, B, U, H = states
        s = K.dot(c_tm1, self.W_h) + self.b_h
        s = K.repeat(s, self.input_length)
        energy = time_distributed_dense(s + H, self.W_a, self.b_a)
        energy = K.squeeze(energy, 2)
        alpha = K.softmax(energy)
        alpha = K.repeat(alpha, self.input_dim)
        alpha = K.permute_dimensions(alpha, (0, 2, 1))
        weighted_H = H * alpha
        v = K.sum(weighted_H, axis=1)
        y, new_states = super(AttentionDecoder, self).step(v, states[:-1])
        return y, new_states

    def call(self, x, mask=None):
        print("AttentionDecoder.call")
        H = x
        x = K.permute_dimensions(H, (1, 0, 2))[-1, :, :]
        if self.stateful or self.state_input or len(self.state_outputs) > 0:
            initial_states = self.states[:]
        else:
            initial_states = self.get_initial_states(H)
        constants = self.get_constants(H) + [H]
        y_0 = x
        x = K.repeat(x, self.output_length)
        initial_states += [y_0]
        last_output, outputs, states = K.rnn(
            self.step,
            x,
            initial_states,
            go_backwards=self.go_backwards,
            mask=mask,
            constants=constants,
            unroll=self.unroll,
            input_length=self.output_length)
        if self.stateful and not self.state_input:
            self.updates = zip(self.states, states)
        self.states_to_transfer = states
        return outputs

    def assert_input_compatibility(self, x):
        shape = x._keras_shape
        assert K.ndim(x) == 3, "AttentionDecoder requires 3D  input, not " + str(K.ndim(x)) + "D."
        assert shape[-1] == self.output_dim or not self.output_dim, "output_dim of AttentionDecoder should be same as the last dimension in the input shape. output_dim = "+ str(self.output_dim) + ", got tensor with shape : " + str(shape) + "."

    def get_output_shape_for(self, input_shape):
        input_shape = list(input_shape)
        output_shape = input_shape[:1] + [self.output_length] + input_shape[2:]
        return tuple(output_shape)

    def get_config(self):
        config = {'name': self.__class__.__name__}
        base_config = super(AttentionDecoder, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

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
        self.add(TimeDistributed(Dense(output_dim, **kwargs)))
        self.encoder = encoder
        self.decoder = decoder


class S2s(Sequential):
    def __init__(self, output_dim, hidden_dim, output_length, depth=1, broadcast_state=True, inner_broadcast_state=True, peek=False, dropout=0.1, **kwargs):
        super(S2s, self).__init__()
        depth = (depth, depth)
        shape = kwargs['batch_input_shape']
        del kwargs['batch_input_shape']

        self.add(LSTMEncoder(batch_input_shape=shape, output_dim=hidden_dim, state_input=False, return_sequences=True, **kwargs))
        self.add(AttentionDecoder(hidden_dim=hidden_dim, output_length=output_length, state_input=broadcast_state, **kwargs))
        self.layers[0].broadcast_state(self.layers[1])
        [self.encoder,self.decoder] = self.layers



class GiveExamples(keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs={}):
        snippet = lambda s: s.tostring()[:50].decode(errors='ignore').replace('\n', '<NL>')
        sample = pat[:3, :]
        chars = model.predict(
            np.eye(128, dtype=np.float32)[sample],
            batch_size=bs
            ).argmax(axis=2).astype(np.uint8)
        print('\n' + 
            '\n'.join(
                [ snippet(st) for st in chars]
                + [snippet(st) for st in sample]))


model = Seq2seq(
    batch_input_shape=(__block_size, __n_chop, n_input),
    hidden_dim=__n_hidden, # can be anything
    #input_length=__n_chop,
    output_length=__n_chop,
    output_dim=n_output,
    activation='softmax',
    #depth=1,
    #consume_less="cpu"
    )
#model.add(keras.layers.core.Activation('softmax'))
#model = keras.models.Sequential([
#    keras.layers.Lambda(
#        lambda x: keras.backend.one_hot(keras.backend.cast(x, 'int16'), n_input),
#        batch_input_shape=(__block_size, __n_chop,),
#        output_shape=(__n_chop, n_input)
#    ),
#    model
#])
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early_stopping = EarlyStopping(monitor='val_loss', patience=2)
checkpoint = ModelCheckpoint("15-weights", save_best_only=True)
eye = np.eye(128)
give_examples = GiveExamples()
model.fit(
    eye[pat[0]],
    np.expand_dims(pat[0], 2),
    batch_size=__block_size,
    callbacks=[early_stopping, checkpoint, give_examples],
    shuffle=True,
    validation_split=0.1,
)
model.save("15-weights")
