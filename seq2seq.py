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
import numpy as np

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda" if USE_CUDA else "cpu")


def calculate_codes(topic_for_code, input_seq_for_code, voc, feature_path, batch_size):
    # batch_size = 64 for training, 1 for chatting
    nmfdict = np.load(feature_path)["feature_names"]
    new_input_seq = torch.zeros(batch_size, len(nmfdict))

    for i in range(batch_size):
        for j in range(len(input_seq_for_code[i])):
            input_seq_words = voc.index2word[input_seq_for_code[i][j].item()]
            for check_index in range(len(nmfdict)):
                if nmfdict[check_index] == input_seq_words:
                    new_input_seq[i][check_index] = 1

    three_d_topic = topic_for_code.repeat(batch_size, 1, 1).to(device)
    three_d_q = new_input_seq.repeat(1, 1, 1).permute(1, 2, 0).to(device)

    return torch.bmm(three_d_topic, three_d_q)



class EncoderRNN(nn.Module):
    def __init__(self,
                 hidden_size,
                 embedding,
                 topics,
                 n_layers=1,
                 dropout=0,
                 batch_size=64):
        super(EncoderRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.embedding = embedding
        self.topics = topics
        self.batch_size = batch_size
        # self.voc = voc
        # Initialize GRU; the input_size and hidden_size params are both set to 'hidden_size'
        #   because our input size is a word embedding with number of features == hidden_size
        self.gru = nn.GRU(hidden_size,
                          hidden_size,
                          n_layers,
                          dropout=(0 if n_layers == 1 else dropout), bidirectional=True)

    def forward(self, input_seq, input_lengths, hidden=None):
        #print(input_seq)
        batch_size = 1
        #when chatting, set to 1
        #get the codes first
        input_seq_for_code = input_seq.transpose(0,1)
        topic_for_code = self.topics
        #input_seq_for_code_np = input_seq_for_code.numpy()
        #print('topic_for_code', topic_for_code.size())
        #print('input_seq_for_code',input_seq_for_code.size())
        #print(input_seq_for_code[0])
        feature_path = r"delta-nmf.npz"
        codes = calculate_codes(topic_for_code, input_seq_for_code, d.voc, feature_path, self.batch_size)
        #print(codes.size())
        # Convert word indexes to embeddings
        embedded = self.embedding(input_seq)
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # Forward pass through GRU
        outputs, hidden = self.gru(packed, hidden)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs)
        # Sum bidirectional GRU outputs
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, : ,self.hidden_size:]
        # Return output and final hidden state
        return outputs, hidden, codes

class Attn(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attn, self).__init__()
        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size, hidden_size)
        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
            self.v = nn.Parameter(torch.FloatTensor(hidden_size))

    def dot_score(self, hidden, encoder_output):
        return torch.sum(hidden * encoder_output, dim=2)

    def general_score(self, hidden, encoder_output):
        energy = self.attn(encoder_output)
        return torch.sum(hidden * energy, dim=2)

    def concat_score(self, hidden, encoder_output):
        energy = self.attn(torch.cat((hidden.expand(encoder_output.size(0), -1, -1), encoder_output), 2)).tanh()
        return torch.sum(self.v * energy, dim=2)

    def forward(self, hidden, encoder_outputs):
        # Calculate the attention weights (energies) based on the given method
        if self.method == 'general':
            attn_energies = self.general_score(hidden, encoder_outputs)
        elif self.method == 'concat':
            attn_energies = self.concat_score(hidden, encoder_outputs)
        elif self.method == 'dot':
            attn_energies = self.dot_score(hidden, encoder_outputs)

        # Transpose max_length and batch_size dimensions
        attn_energies = attn_energies.t()

        # Return the softmax normalized probability scores (with added dimension)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

# Message attention
class LuongAttnDecoderRNN(nn.Module):
    def __init__(self, attn_model, embedding, hidden_size, output_size, n_layers=1, dropout=0.1):
        super(LuongAttnDecoderRNN, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.embedding = embedding
        self.embedding_dropout = nn.Dropout(dropout)
        self.gru = nn.GRU(hidden_size, hidden_size, n_layers, dropout=(0 if n_layers == 1 else dropout))
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.attn = Attn(attn_model, hidden_size)

    def forward(self, input_step, last_hidden, encoder_outputs):
        # Note: we run this one step (word) at a time
        # Get embedding of current input word
        embedded = self.embedding(input_step)
        embedded = self.embedding_dropout(embedded)
        # Forward through unidirectional GRU
        rnn_output, hidden = self.gru(embedded, last_hidden)
        # Calculate attention weights from the current GRU output
        attn_weights = self.attn(rnn_output, encoder_outputs)
        # Multiply attention weights to encoder outputs to get new "weighted sum" context vector
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))
        # Concatenate weighted context vector and GRU output using Luong eq. 5 (our eq. 17)
        rnn_output = rnn_output.squeeze(0)
        context = context.squeeze(1)
        concat_input = torch.cat((rnn_output, context), 1)
        # print(concat_input.size())
        concat_output = torch.tanh(self.concat(concat_input))
        # Predict next word using Luong eq. 6 (our eq. 19 third eq. only with the first term)
        output = self.out(concat_output)
        output = F.softmax(output, dim=1)
        # Return output and final hidden state
        return output, hidden


def maskNLLLoss(inp, target, mask):
    nTotal = mask.sum()
    crossEntropy = -torch.log(torch.gather(inp, 1, target.view(-1, 1)).squeeze(1))
    loss = crossEntropy.masked_select(mask).mean()
    loss = loss.to(device)
    return loss, nTotal.item()


def validation(input_variable, lengths, target_variable, mask, max_target_len, encoder, decoder, embedding,
          encoder_optimizer, decoder_optimizer, batch_size, clip, max_length=d.MAX_LENGTH):
    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden, codes = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[d.SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    teacher_forcing_ratio = 1.0
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, codes, batch_size
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, codes, batch_size
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    return sum(print_losses) / n_totals

def train(input_variable,
          lengths,
          target_variable,
          mask,
          max_target_len,
          encoder,
          decoder,
          embedding,
          encoder_optimizer,
          decoder_optimizer,
          batch_size,
          clip,
          max_length=d.MAX_LENGTH):

    # Zero gradients
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    # Set device options
    input_variable = input_variable.to(device)
    lengths = lengths.to(device)
    target_variable = target_variable.to(device)
    mask = mask.to(device)

    # Initialize variables
    loss = 0
    print_losses = []
    n_totals = 0

    # Forward pass through encoder
    encoder_outputs, encoder_hidden, codes = encoder(input_variable, lengths)

    # Create initial decoder input (start with SOS tokens for each sentence)
    decoder_input = torch.LongTensor([[d.SOS_token for _ in range(batch_size)]])
    decoder_input = decoder_input.to(device)

    # Set initial decoder hidden state to the encoder's final hidden state
    #decoder_hidden = encoder_hidden[:decoder.n_layers]
    decoder_hidden = encoder_hidden[:decoder.n_layers]

    # Determine if we are using teacher forcing this iteration
    teacher_forcing_ratio = 1.0
    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    # Forward batch of sequences through decoder one time step at a time
    if use_teacher_forcing:
        for t in range(max_target_len):

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, codes, batch_size
            )
            # Teacher forcing: next input is current target
            decoder_input = target_variable[t].view(1, -1)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal
    else:
        for t in range(max_target_len):
            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, codes, batch_size
            )
            # No teacher forcing: next input is decoder's own current output
            _, topi = decoder_output.topk(1)
            decoder_input = torch.LongTensor([[topi[i][0] for i in range(batch_size)]])
            decoder_input = decoder_input.to(device)
            # Calculate and accumulate loss
            mask_loss, nTotal = maskNLLLoss(decoder_output, target_variable[t], mask[t])
            loss += mask_loss
            print_losses.append(mask_loss.item() * nTotal)
            n_totals += nTotal

    # Perform backpropatation
    loss.backward()

    # Clip gradients: gradients are modified in place
    _ = nn.utils.clip_grad_norm_(encoder.parameters(), clip)
    _ = nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Adjust model weights
    encoder_optimizer.step()
    decoder_optimizer.step()

    return sum(print_losses) / n_totals


def trainIters(model_name,
               voc,
               voc_validation,
               pairs,
               pairs_validation,
               encoder,
               decoder,
               encoder_optimizer,
               decoder_optimizer,
               embedding,
               encoder_n_layers,
               decoder_n_layers,
               save_dir,
               n_iteration,
               batch_size,
               print_every,
               save_every,
               clip,
               corpus_name,
               loadFilename,
               DICT_NAME,
               checkpoint):

    #history record file
    history_file = open(r'history_delta.txt', 'w')

    # Load batches for each iteration
    training_batches = [p.batch2TrainData(voc, [random.choice(pairs) for _ in range(batch_size)])
                      for _ in range(n_iteration)]
    training_batches_validation = [p.batch2TrainData(voc_validation, [random.choice(pairs_validation) for _ in range(batch_size)])
                        for _ in range(n_iteration)]

    # Initializations
    print('Initializing ...')
    start_iteration = 1
    print_loss = 0
    print_loss_validation = 0
    if loadFilename:
        start_iteration = checkpoint['iteration'] + 1

    # Training loop
    print("Training...")
    for iteration in range(start_iteration, n_iteration + 1):
        training_batch = training_batches[iteration - 1]
        training_batch_validation = training_batches_validation[iteration - 1]

        # Extract fields from batch
        input_variable, lengths, target_variable, mask, max_target_len = training_batch
        input_variable_validation, lengths_validation, target_variable_validation, mask_validation, max_target_len_validation = training_batch_validation

        # Run a training iteration with batch
        loss = train(input_variable, lengths, target_variable, mask, max_target_len, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
        print_loss += loss
        loss_validation = validation(input_variable_validation, lengths_validation, target_variable_validation, mask_validation, max_target_len_validation, encoder,
                     decoder, embedding, encoder_optimizer, decoder_optimizer, batch_size, clip)
        print_loss_validation += loss_validation

        # Print progress
        if iteration % print_every == 0:
            print_loss_avg = print_loss / print_every
            print_loss_avg_validation = print_loss_validation / print_every
            print("Iteration: {}; Percent complete: {:.1f}%; Training loss: {:.4f}; Validation loss: {:.4f}".format(iteration, iteration / n_iteration * 100, print_loss_avg, print_loss_avg_validation))
            history_file.write('{}'.format(print_loss_avg))
            history_file.write(' ')
            history_file.write('{}'.format(print_loss_avg_validation))
            history_file.write("\n")

            print_loss = 0
            print_loss_validation = 0

        # Save checkpoint
        if (iteration % save_every == 0):
            hidden_size = 500
            directory = os.path.join(save_dir, model_name, corpus_name, '{}-{}_{}'.format(encoder_n_layers, decoder_n_layers, hidden_size))
            if not os.path.exists(directory):
                os.makedirs(directory)
            torch.save({
                'iteration': iteration,

            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpointiteration_'+DICT_NAME)))

            torch.save({
                #'iteration': iteration,
                'en': encoder.state_dict(),

            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpointencoder_'+DICT_NAME)))

            torch.save({


                'de': decoder.state_dict(),

            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpointdecoder_'+DICT_NAME)))

            torch.save({


                'en_opt': encoder_optimizer.state_dict(),

            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpointenopt_'+DICT_NAME)))

            torch.save({


                'de_opt': decoder_optimizer.state_dict(),

            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpointdeopt_'+DICT_NAME)))

            torch.save({


                'loss': loss,

            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpointloss_'+DICT_NAME)))

            torch.save({


                'voc_dict': voc.__dict__,

            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpointvocdict_'+DICT_NAME)))

            torch.save({

                'embedding': embedding.state_dict()
            }, os.path.join(directory, '{}_{}.tar'.format(iteration, 'checkpointembbedding_'+DICT_NAME)))

    history_file.close()

class GreedySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(GreedySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, input_seq, input_length, max_length, batch_size):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden, codes = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * d.SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input,
                                                          decoder_hidden,
                                                          encoder_outputs,
                                                          codes,
                                                          batch_size)
            # Obtain most likely word token and its softmax score
            decoder_scores, decoder_input = torch.max(decoder_output, dim=1)
            #print(decoder_scores)
            #print(decoder_input)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores

def my_random_pick(probabilities):
    x=random.uniform(0,1)
    #print(probabilities[0][20])
    cumulative_probability = 0.0
    #list = [n for n in range(len(probabilities))]
    for item in range(len(probabilities[0])):
        cumulative_probability += probabilities[0][item]
        if x < cumulative_probability:
            break
    item_list = torch.zeros([1], device=device, dtype=torch.long)
    item_list[0] = item
    score = torch.zeros([1],device=device)
    score[0]=probabilities[0][item]
    #print(item_list)
    #print(score)
    #print("one random pick")

    return item_list, score

class ProbabilitySearchDecoder(nn.Module):
    def __init__(self, encoder, decoder, batch_size=1):
        super(ProbabilitySearchDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.batch_size = batch_size

    def forward(self, input_seq, input_length, max_length):
        # Forward input through encoder model
        encoder_outputs, encoder_hidden, codes = self.encoder(input_seq, input_length)
        # Prepare encoder's final hidden layer to be first hidden input to the decoder
        decoder_hidden = encoder_hidden[:self.decoder.n_layers]
        # Initialize decoder input with SOS_token
        decoder_input = torch.ones(1, 1, device=device, dtype=torch.long) * d.SOS_token
        # Initialize tensors to append decoded words to
        all_tokens = torch.zeros([0], device=device, dtype=torch.long)
        all_scores = torch.zeros([0], device=device)
        # Iteratively decode one word token at a time
        for _ in range(max_length):
            # Forward pass through decoder
            decoder_output, decoder_hidden = self.decoder(decoder_input,
                                                          decoder_hidden,
                                                          encoder_outputs,
                                                          codes,
                                                          self.batch_size)
            # Obtain word based on probability distribution token and its softmax score
            decoder_input, decoder_scores = my_random_pick(decoder_output)
            # Record token and score
            all_tokens = torch.cat((all_tokens, decoder_input), dim=0)
            all_scores = torch.cat((all_scores, decoder_scores), dim=0)
            # Prepare current token to be next decoder input (add a dimension)
            decoder_input = torch.unsqueeze(decoder_input, 0)
        # Return collections of word tokens and scores
        return all_tokens, all_scores


def evaluate(encoder, decoder, searcher, voc, sentence, max_length=d.MAX_LENGTH):
    ### Format input sentence as a batch
    # words -> indexes
    indexes_batch = [p.indexesFromSentence(voc, sentence)]
    # Create lengths tensor
    lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
    # Transpose dimensions of batch to match models' expectations
    input_batch = torch.LongTensor(indexes_batch).transpose(0, 1)
    # Use appropriate device
    input_batch = input_batch.to(device)
    lengths = lengths.to(device)
    # Decode sentence with searcher
    tokens, scores = searcher(input_batch, lengths, max_length)
    # indexes -> words
    decoded_words = [voc.index2word[token.item()] for token in tokens]
    return decoded_words


def evaluateInput(encoder, decoder, searcher, voc):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = d.normalizeString(input_sentence)
            # Evaluate sentence
            start_time = time.time()
            output_words = evaluate(encoder, decoder, searcher, voc, input_sentence)
            # Format and print response sentence
            output_words[:] = [x for x in output_words if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words))
            end_time = time.time()
            print('Responding time:', end_time - start_time)
        except KeyError:
            print("Error: Encountered unknown word.")

def multi_evaluateInput(encoder1, decoder1, encoder2, decoder2, searcher1, searcher2, voc):
    input_sentence = ''
    while(1):
        try:
            # Get input sentence
            input_sentence = input('> ')
            # Check if it is quit case
            if input_sentence == 'q' or input_sentence == 'quit': break
            # Normalize sentence
            input_sentence = d.normalizeString(input_sentence)
            # Evaluate sentence
            #start_time = time.time()
            output_words1 = evaluate(encoder1, decoder1, searcher1, voc, input_sentence)
            # Format and print response sentence
            output_words1[:] = [x for x in output_words1 if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words1))
            output_words2 = evaluate(encoder2, decoder2, searcher2, voc, input_sentence)
            output_words2[:] = [x for x in output_words2 if not (x == 'EOS' or x == 'PAD')]
            print('Bot:', ' '.join(output_words2))
            #end_time = time.time()
            #print('Responding time:', end_time - start_time)
        except KeyError:
            print("Error: Encountered unknown word.")
