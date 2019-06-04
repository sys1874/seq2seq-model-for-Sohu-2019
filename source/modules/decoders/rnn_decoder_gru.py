#!/usr/bin/env python
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/decoders/rnn_decoder.py
"""

import torch
import torch.nn as nn

from source.modules.attention import Attention
from source.modules.decoders.state import DecoderState
from source.utils.misc import Pack
from source.utils.misc import sequence_mask


class RNNDecoder(nn.Module):
    """
    A GRU recurrent neural network decoder.
    """
    def __init__(self,
                 input_size,
                 hidden_size,
                 embedder=None,
                 num_layers=1,
                 attn_mode=None,
                 memory_size=None,
                 dropout=0.0):
        super(RNNDecoder, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedder = embedder
        self.num_layers = num_layers
        self.attn_mode = None if attn_mode == 'none' else attn_mode
        self.memory_size = memory_size or hidden_size
        self.dropout = dropout
        self.out_input_size = hidden_size
        self.class_num=3
        self.rnn_input_size=input_size+hidden_size


        if self.attn_mode is not None:
            self.attention = Attention(query_size=self.hidden_size,
                                       memory_size=self.memory_size,
                                       hidden_size=self.memory_size,
                                       mode=self.attn_mode,
                                       project=False)

        self.rnn = nn.GRU(input_size=self.rnn_input_size,
                          hidden_size=self.hidden_size,
                          num_layers=self.num_layers,
                          dropout=self.dropout if self.num_layers > 1 else 0,
                          batch_first=True)

        self.fc1=nn.Linear(2*self.hidden_size, self.hidden_size)

        self.output_layer = nn.Sequential(
                nn.Linear(self.out_input_size, self.out_input_size//2),
                nn.Dropout(p=self.dropout),
                nn.ReLU(),
                nn.Linear(self.out_input_size//2, self.class_num),
                nn.Softmax(dim=-1)
            )

        self.fusion=nn.Sequential(
            nn.Linear(2*self.hidden_size,1),
            nn.Sigmoid()
        )

    def initialize_state(self,
                         hidden,
                         attn_memory=None,
                         input_feed=None,
                         mask=None):
        """
        initialize_state
        """

        if self.attn_mode is not None:
            assert attn_memory is not None



        init_state = DecoderState(
            hidden=hidden,
            attn_memory=attn_memory,
            input_feed=input_feed,
            mask=mask
        )
        return init_state

    def decode(self, input, state, is_training=False):
        """
        decode
        """
        hidden = state.hidden
        input_feed= state.input_feed
        output = Pack()

        if self.embedder is not None:
            input = self.embedder(input)

        # shape: (batch_size, 1, input_size)
        input = input.unsqueeze(1)
        input_feed = input_feed
        rnn_input = torch.cat([input,input_feed] ,dim=-1)
        rnn_output, new_hidden = self.rnn(rnn_input, hidden)



        attn_memory = state.attn_memory
        query = new_hidden[-1].unsqueeze(1)
        weighted_context, attn = self.attention(query=query,
                                                    memory=attn_memory,
                                                    mask=state.mask.eq(0))


        final_output=torch.cat([query, weighted_context], dim=-1)

        # fusion_sigmod=self.fusion(final_output)
        #
        # fusion_hidden=fusion_sigmod*weighted_context+(1-fusion_sigmod)*query

        final_output=self.fc1(final_output)

        output.add(attn=attn)


        state.hidden = new_hidden
        state.input_feed=final_output
        # state.input_feed=fusion_hidden
        out_input=final_output

        if is_training:
            return out_input, state, output
        else:
            log_prob = self.output_layer(out_input)
            return log_prob, state, output

    def forward(self, num_target_input, state):
        """
        forward
        """
        inputs, lengths = num_target_input
        batch_size, max_len = inputs.size()
        vocab_size=state.attn_memory.size(1)

        out_inputs = inputs.new_zeros(
            size=(batch_size, max_len, self.out_input_size),
            dtype=torch.float)
        # batch_size  x  max_len x out_input_size
        out_copy = inputs.new_zeros(size=(batch_size, max_len, vocab_size), dtype=torch.float)
        # sort by lengths
        sorted_lengths, indices = lengths.sort(descending=True)
        inputs = inputs.index_select(0, indices)
        state = state.index_select(indices)

        # number of valid input (i.e. not padding index) in each time step
        num_valid_list = sequence_mask(sorted_lengths).int().sum(dim=0)

        for i, num_valid in enumerate(num_valid_list):
            dec_input = inputs[:num_valid, i]
            valid_state = state.slice_select(num_valid)
            out_input, valid_state, outputs = self.decode(
                dec_input, valid_state, is_training=True)
            state.hidden[:, :num_valid] = valid_state.hidden
            # state.hidden[1][:, :num_valid] = valid_state.hidden[1]
            out_inputs[:num_valid, i] = out_input.squeeze(1)
            out_copy[:num_valid, i] = outputs.attn.squeeze(1)
        # Resort
        _, inv_indices = indices.sort()
        state = state.index_select(inv_indices)
        out_inputs = out_inputs.index_select(0, inv_indices)
        out_copy = out_copy.index_select(0, inv_indices)

        log_probs = self.output_layer(out_inputs)
        return log_probs, state, out_copy
