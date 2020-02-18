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

import seq2seq

import prepare_data_for_model as p
import load_trim_data as d
import numpy as np
import ta_seq2seq

'''
Chat with the trained chatbot with desired NMF-topic filter 
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
batch_size = 1
checkpoint_loading_from = 100

path = 'Data/save/topiccb_model/cornell movie-dialogs corpus/1-1_500/' + str(checkpoint_loading_from) + '_'

### Choose desired NMF-topic filter
DICT_NAME = 'delta'
# DICT_NAME = 'news'
# DICT_NAME = 'shakes'

DICT_PATH = DICT_NAME + '-nmf.npz'

# Set checkpoint to load from; set to None if starting from scratch
loadFilename_encoder = path +"checkpointencoder_" + DICT_NAME + '.tar'
loadFilename_decoder = path + "checkpointdecoder_" + DICT_NAME + '.tar'
loadFilename_enopt = path + "checkpointenopt_" + DICT_NAME + '.tar'
loadFilename_deopt = path + "checkpointdeopt_" + DICT_NAME + '.tar'
loadFilename_loss = path + "checkpointloss_" + DICT_NAME + '.tar'
loadFilename_vocdict = path + "checkpointvocdict_" + DICT_NAME + '.tar'
loadFilename_embedding = path + "checkpointembbedding_" + DICT_NAME + '.tar'
checkpoint_iter = 64000
#loadFilename = os.path.join(save_dir, model_name, corpus_name,
#                            '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size),
#                            '{}_checkpoint.tar'.format(checkpoint_iter))


# Load model if a loadFilename is provided
if loadFilename_encoder:
    # If loading on same machine the model was trained on
    checkpoint_encoder = torch.load(loadFilename_encoder)
    checkpoint_decoder = torch.load(loadFilename_decoder)
    checkpoint_enopt = torch.load(loadFilename_enopt)
    checkpoint_deopt = torch.load(loadFilename_deopt)
    checkpoint_loss = torch.load(loadFilename_loss)
    checkpoint_vocdict = torch.load(loadFilename_vocdict)
    checkpoint_embedding = torch.load(loadFilename_embedding)

    # If loading a model trained on GPU to CPU
    #checkpoint = torch.load(loadFilename, map_location=torch.device('cpu'))
    encoder_sd = checkpoint_encoder['en']
    decoder_sd = checkpoint_decoder['de']
    # print('decoder_sd',  decoder_sd.size())
    encoder_optimizer_sd = checkpoint_enopt['en_opt']
    decoder_optimizer_sd = checkpoint_deopt['de_opt']
    embedding_sd = checkpoint_embedding['embedding']
    d.voc.__dict__ = checkpoint_vocdict['voc_dict']
    #print("not empty")


print('Building encoder and decoder ...')
# Initialize word embeddings
embedding = nn.Embedding(d.voc.num_words, hidden_size)
if loadFilename_encoder:
    embedding.load_state_dict(embedding_sd)

# DICT_PATH = 'topics.txt'
# topic_dict = torch.tensor(np.loadtxt(DICT_PATH), dtype=torch.float).to(seq2seq.device)

DICT_PATH = 'delta-nmf.npz'
topic_dict = torch.tensor(np.load(DICT_PATH)["dictionary"], dtype=torch.float).to(seq2seq.device)

# Initialize encoder & decoder models
encoder = seq2seq.EncoderRNN(hidden_size=hidden_size,
                             embedding=embedding,
                             topics=topic_dict,
                             n_layers=encoder_n_layers,
                             dropout=dropout,
                             batch_size=batch_size)

#decoder = seq2seq.LuongAttnDecoderRNN(attn_model, embedding, hidden_size, d.voc.num_words, decoder_n_layers, dropout)

enc_hid_dim, dec_hid_dim, emb_dim = hidden_size, hidden_size, hidden_size
decoder = ta_seq2seq.TopicDecoder(attn_model=attn_model,
                                  embedding=embedding,
                                  hidden_size=hidden_size,
                                  output_size=d.voc.num_words,
                                  enc_hid_dim=enc_hid_dim,
                                  dec_hid_dim=dec_hid_dim,
                                  topics=topic_dict,
                                  topic_vocab_size=topic_dict.shape[1],
                                  n_layers=decoder_n_layers,
                                  dropout=dropout,
                                  batch_size=batch_size)


if loadFilename_encoder:
    encoder.load_state_dict(encoder_sd)
    decoder.load_state_dict(decoder_sd)
# Use appropriate device
encoder = encoder.to(device)
decoder = decoder.to(device)
print('Models built and ready to go!')

encoder.eval()
decoder.eval()
searcher = seq2seq.ProbabilitySearchDecoder(encoder, decoder)
seq2seq.evaluateInput(encoder, decoder, searcher, d.voc)










