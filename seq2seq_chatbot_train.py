from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
from torch.jit import script, trace
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import csv
import random
import re
import os
import unicodedata
import codecs
from io import open
import itertools
import math

import load_trim_data as d
import seq2seq
import ta_seq2seq

import numpy as np

'''
Choose desired NMF-topic filter and then train the chatbot over Cornell movie dialogue 
'''


USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


# Configure models
model_name = 'topiccb_model'
attn_model = 'dot'
#attn_model = 'general'
#attn_model = 'concat'
hidden_size = 500
encoder_n_layers = 1
decoder_n_layers = 1
dropout = 0.1
batch_size = 64 # 64 for training, 1 for chatting

### Choose which NMF-learned dictionary you want to use in training
# DICT_NAME = 'delta'
DICT_NAME = 'news'
# DICT_NAME = 'shakes'

DICT_PATH = DICT_NAME + '-nmf.npz'
topic_dict = torch.tensor(np.load(DICT_PATH)["dictionary"], dtype=torch.float).to(seq2seq.device)

# Set checkpoint to load from; set to None if starting from scratch
loadFilename = None
checkpoint_iter = 64000
#loadFilename = os.path.join(save_dir, model_name, corpus_name,
#                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                            '{}_checkpoint.tar'.format(checkpoint_iter))


# Load model if a loadFilename is provided
if loadFilename:
    # If loading on same machine the model was trained on
    checkpoint = torch.load(loadFilename)
    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint['en']
    decoder_sd = checkpoint['de']
    encoder_optimizer_sd = checkpoint['en_opt']
    decoder_optimizer_sd = checkpoint['de_opt']
    embedding_sd = checkpoint['embedding']
    voc__dict__ = checkpoint['voc_dict']


print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(d.voc.num_words, hidden_size)
if loadFilename:
    embedding.load_state_dict(embedding_sd)

# Initialize encoder & decoder models
encoder = seq2seq.EncoderRNN(hidden_size=hidden_size,
                             embedding=embedding,
                             topics=topic_dict,
                             n_layers=encoder_n_layers,
                             dropout=dropout,
                             batch_size=batch_size)


#decoder = seq2seq.LuongAttnDecoderRNN(attn_model, embedding, hidden_size, d.voc.num_words, decoder_n_layers, dropout)
enc_hid_dim, dec_hid_dim, emb_dim = hidden_size, hidden_size, hidden_size
#voc_dim = d.voc.num_words
#attention = seq2seq.Attn(attn_model, hidden_size)
#ta_attn = ta_seq2seq.TopicAttention(topic_dict.shape[1], enc_hid_dim, dec_hid_dim)
decoder = ta_seq2seq.TopicDecoder(attn_model,
                                  embedding,
                                  hidden_size,
                                  d.voc.num_words,
                                  enc_hid_dim,
                                  dec_hid_dim,
                                  topic_dict,
                                  topic_dict.shape[1],
                                  decoder_n_layers,
                                  dropout)
if loadFilename:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)

# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')


#start training
# Configure training/optimization
clip = 50.0
teacher_forcing_ratio = 1.0
learning_rate = 0.0001
decoder_learning_ratio = 5.0
n_iteration = 64000
print_every = 1
save_every = 100

# Ensure dropout layers are in train mode
encoder.train()
decoder.train()

# Initialize optimizers
print('Building optimizers ...')
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
if loadFilename:
    encoder_optimizer.load_state_dict(encoder_optimizer_sd)
    decoder_optimizer.load_state_dict(decoder_optimizer_sd)

# If you have cuda, configure cuda to call
for state in encoder_optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

for state in decoder_optimizer.state.values():
    for k, v in state.items():
        if isinstance(v, torch.Tensor):
            state[k] = v.cuda()

# Run training iterations
print("Starting Training!")
seq2seq.trainIters(model_name=model_name,
                   voc=d.voc,
                   voc_validation=d.voc_validation,
                   pairs=d.pairs,
                   pairs_validation=d.pairs_validation,
                   encoder=encoder,
                   decoder=decoder,
                   encoder_optimizer=encoder_optimizer,
                   decoder_optimizer=decoder_optimizer,
                   embedding=embedding,
                   encoder_n_layers=encoder_n_layers,
                   decoder_n_layers=decoder_n_layers,
                   save_dir=d.save_dir,
                   n_iteration=n_iteration,
                   batch_size=batch_size,
                   print_every=print_every,
                   save_every=save_every,
                   clip=clip,
                   corpus_name=d.corpus_name,
                   loadFilename=loadFilename,
                   DICT_NAME=DICT_NAME,
                   checkpoint=None)