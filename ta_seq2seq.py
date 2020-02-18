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
import time
from torch.utils.data.sampler import  WeightedRandomSampler

import load_trim_data as d
import prepare_data_for_model as p
import seq2seq

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


class TopicAttention(nn.Module):
    def __init__(self, topic_vocab_size, enc_hid_dim, dec_hid_dim):
        super(TopicAttention, self).__init__()
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim

        self.attn = nn.Linear(topic_vocab_size + dec_hid_dim + enc_hid_dim, dec_hid_dim)
        self.v = nn.Parameter(torch.rand(dec_hid_dim))

    def forward(self, hidden, topic_dict, enc_hidden):
        # hidden = [batch_size, dec_hid_dim]
        # topic_dict = [num_topics, topic_vocab_size]
        # enc_hidden = [batch_size, enc_hid_dim * 2]
        batch_size = enc_hidden.shape[0]
        num_topics = topic_dict.shape[0]

        #print('hidden', hidden.size())
        #print('topic_dict', topic_dict.size())
        #print('enc_hidden', enc_hidden.size())
        hidden = hidden.repeat(num_topics, 1, 1).permute(1, 0, 2)
        enc_hidden = enc_hidden.repeat(num_topics, 1, 1).permute(1, 0, 2)
        topic_dict = topic_dict.repeat(batch_size, 1, 1)

        # hidden = [batch_size, num_topics, dec_hid_dim]
        # enc_hidden = [batch_size, num_topics, 2 * enc_hid_dim]
        # topic_dict = [batch_size, num_topics, topic_vocab_size]
        #print(num_topics)
        #print('hidden',hidden.size())
        #print('topic_dict',topic_dict.size())
        #print('enc_hidden',enc_hidden.size())
        #print(torch.cat((hidden, topic_dict, enc_hidden), dim=2).size())
        energy = torch.tanh(self.attn(torch.cat((hidden, topic_dict, enc_hidden), dim=2)))

        # energy = [batch_size, dec_hid_dim]
        energy = energy.permute(0, 2, 1)

        v = self.v.repeat(batch_size, 1).unsqueeze(1)

        # v = [batch_size, 1, dec_hid_dim]
        attention = torch.bmm(v, energy).squeeze(1)

        return F.softmax(attention, dim=1).unsqueeze(1)


class TopicDecoder(nn.Module):
    def __init__(self,
                 attn_model,
                 embedding,
                 hidden_size,
                 output_size,
                 enc_hid_dim,
                 dec_hid_dim,
                 topics,
                 topic_vocab_size,
                 n_layers=1,
                 dropout=0.1,
                 batch_size=1):
        super(TopicDecoder, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.topics = topics
        self.topic_vocab_size = topic_vocab_size
        self.batch_size = batch_size

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2 + topic_vocab_size + 1000, hidden_size) #1000 is len of topic dictionary
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = seq2seq.Attn(attn_model, hidden_size)
        self.topic_attn = TopicAttention(topic_vocab_size, enc_hid_dim, dec_hid_dim)

    def forward(self,
                input_step,
                last_hidden,
                encoder_outputs,
                codes,
                batch_size=1):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        # print('encoder_outputs',encoder_outputs.size())
        # batch_size = 64  #for chat : set to 1
        # print('batch_size', batch_size)
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)

        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # print('last_hidden', last_hidden.size())
        # print('self.topics', self.topics.size())
        # print('encoder_outputs', encoder_outputs.size())
        topic_attn_weights = self.topic_attn(last_hidden, self.topics, encoder_outputs[-1])

        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        encoder_outputs = encoder_outputs.permute(1,0,2)
        # print('attn_weights', attn_weights.size())
        # print('encoder_outputs', encoder_outputs.size())

        context = torch.bmm(attn_weights, encoder_outputs)
        # print('context', context.size())
        # print('topic_attn_weights',topic_attn_weights.size())
        # print('self.topics.repeat(batch_size, 1, 1)',self.topics.repeat(batch_size, 1, 1).size())
        topic_context = torch.bmm(topic_attn_weights, self.topics.repeat(batch_size, 1, 1))
        # context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # print('self.topics',self.topics.size())
        # print('codes',codes.size())
        topic_for_Pk = self.topics.repeat(batch_size,1,1).permute(0,2,1)
        # print('topic_for_Pk',topic_for_Pk.size())
        Pk_context = torch.bmm(topic_for_Pk, codes)
        # print('Pk_context',Pk_context.size())

        # Concatenate weighted context vector and GRU output using Luong eq. 5
        # print('rnn_output', rnn_output.size())
        # print('context', context.size())
        # print('topic_context', topic_context.size())
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        topic_context = topic_context.squeeze(1)
        # print('rnn_output', rnn_output.size())
        # print('context', context.size())
        # print('topic_context', topic_context.size())
        Pk_context = Pk_context.squeeze(2)
        concat_input = torch.cat((rnn_output, context, topic_context, Pk_context), 1)
        # print('concat_input', concat_input.size())
        concat_output = torch.tanh(self.concat(concat_input))

        # Predict next word using Luong eq. 6
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden




class error_TopicDecoder(nn.Module):
    def __init__(self,
                 output_dim,
                 emb_dim,
                 enc_hid_dim,
                 dec_hid_dim,
                 dropout,
                 attention,
                 topic_attention,
                 topics,
                 embedding):
        super(TopicDecoder, self).__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.topic_dim = topics.shape[1]
        self.dropout = dropout
        self.attention = attention
        self.topic_attention = topic_attention
        self.topics = topics

        self.embedding = embedding

        self.rnn = nn.GRU(emb_dim + 2 * enc_hid_dim + self.topic_dim, dec_hid_dim)
        self.out = nn.Linear(emb_dim + dec_hid_dim + 2 * enc_hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, word, hidden, enc_out):
        # word = [batch_size]
        # hidden = [batch_size, dec_hid_dim]
        # enc_out = [sent_len, batch_size, 2 * enc_hid_dim]
        # why did i use these dimensions for enc_out?
        batch_size = word.shape[0]

        word = word.unsqueeze(0)
        embedded = self.embedding(word)
        embedded = self.embedding_dropout(embedded)
        embedded = self.dropout(self.embedding(word))

        # embedded = [1, batch_size, emb_dim]
        a = self.attention(hidden, enc_out)
        ta = self.topic_attention(hidden, self.topics, enc_out[-1])

        # a = [batch_size, sent_len]
        # ta = [batch_size, num_topics]
        a = a.unsqueeze(1)
        ta = ta.unsqueeze(1)

        enc_out = enc_out.permute(1, 0, 2)

        # enc_out = [batch_size, sent_len, 2 * enc_hid_dim]
        weighted = torch.bmm(a, enc_out)
        topics_weighted = torch.bmm(ta, self.topics.repeat(batch_size, 1, 1))

        # weighted = [batch_size, 1, 2 * enc_hid_dim]
        # topics_weighted = [batch_size, 1,  topic_dim]
        # leave these unsqueezed so the RNN treats it as a seq of len 1
        weighted = weighted.permute(1, 0, 2)
        topics_weighted = topics_weighted.permute(1, 0, 2)

        rnn_input = torch.cat((embedded, weighted, topics_weighted), dim=2)

        output, hidden = self.rnn(rnn_input, hidden.unsqueeze(0))

        embedded = embedded.squeeze(0)
        output = output.squeeze(0)
        weighted = weighted.squeeze(0)

        # use topics_weighted in the final linear layer?
        output = self.out(torch.cat((output, weighted, embedded), dim=1))

        # return weighted makes this only work with biased seq2seq, make cleaner later
        return output, hidden.squeeze(0)#, weighted
