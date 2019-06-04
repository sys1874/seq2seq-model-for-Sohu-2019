#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/models/seq2seq.py
"""

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_

from source.models.base_model import BaseModel
from source.modules.embedder import Embedder
from source.modules.encoders.rnn_encoder_lstm import RNNEncoder
# from source.modules.encoders.rnn_encoder_lstm_pos import RNNEncoder_pos
from source.modules.encoders.rnn_encoder_gru_pos import RNNEncoder_pos
from source.modules.decoders.rnn_decoder_gru import RNNDecoder
from source.utils.criterions import NLLLoss
from source.utils.metrics import accuracy
from source.utils.misc import Pack
from source.utils.misc import sequence_mask
class Entity_Seq2Seq_pos_gru(BaseModel):
    """
    Seq2Seq
    """
    def __init__(self,
                 src_vocab_size,
                 pos_vocab_size,
                 embed_size,
                 hidden_size,
                 padding_idx=None,
                 num_layers=1,
                 bidirectional=True,
                 attn_mode="mlp",
                 with_bridge=False,
                 dropout=0.0,
                 use_gpu=False,
                 pretrain_epoch=5,
                 ):
        super(Entity_Seq2Seq_pos_gru, self).__init__()

        self.src_vocab_size = src_vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.padding_idx = padding_idx
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.attn_mode = attn_mode
        self.with_bridge = with_bridge
        self.dropout = dropout
        self.use_gpu = use_gpu
        self.pretrain_epoch=pretrain_epoch
        self.pos_vocab_size=pos_vocab_size

        enc_embedder = Embedder(num_embeddings=self.src_vocab_size,
                                embedding_dim=self.embed_size,
                                padding_idx=self.padding_idx)
        pos_embedder = Embedder(num_embeddings=self.pos_vocab_size,
                                embedding_dim=50,
                                padding_idx=self.padding_idx)
        self.encoder = RNNEncoder_pos(input_size=self.embed_size+50,
                                  hidden_size=self.hidden_size,
                                  embedder=enc_embedder,
                                  pos_embedder=pos_embedder,
                                  num_layers=self.num_layers,
                                  bidirectional=self.bidirectional,
                                  dropout=self.dropout)

        if self.with_bridge:
            self.bridge1 = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Tanh(),
            )
            self.bridge2 = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.Tanh(),
            )



        self.decoder = RNNDecoder(input_size=self.embed_size,
                                  hidden_size=self.hidden_size,
                                  embedder=enc_embedder,
                                  num_layers=self.num_layers,
                                  attn_mode=self.attn_mode,
                                  memory_size=self.hidden_size,
                                  dropout=self.dropout)

        # Loss Definition
        if self.use_gpu:
            self.cuda()

    def encode(self, inputs, hidden=None):
        """
        encode
        """
        outputs = Pack()
        enc_inputs, lengths = inputs.num_src
        pos_inputs = inputs.num_pos[0]
        enc_outputs, enc_hidden = self.encoder(enc_inputs, pos_inputs, hidden)

        if self.with_bridge:
            enc_hidden = self.bridge1(enc_hidden)


        layer, batch_size, dim=enc_hidden.size()
        dec_init_state = self.decoder.initialize_state(
            hidden=enc_hidden,
            input_feed=enc_hidden.data.new(batch_size,dim).zero_() \
                              .unsqueeze(1),
            attn_memory=enc_outputs if self.attn_mode else None,
            mask= inputs.mask[0])
        return outputs, dec_init_state

    def decode(self, input, state):
        """
        decode
        step by step
        """
        log_prob, state, output = self.decoder.decode(input, state)
        return log_prob, state, output

    def forward(self, enc_inputs, dec_inputs, hidden=None):
        """
        forward

        """
        outputs, dec_init_state = self.encode(enc_inputs, hidden)
        log_probs, state, out_copy = self.decoder(dec_inputs, dec_init_state)
        outputs.add(logits=log_probs)
        outputs.add(out_copy=out_copy)
        return outputs

    def collect_metrics(self, outputs, target, emo_target):
        """
        collect_metrics
        """
        num_samples = target[0].size(0)
        num_words = target[1].sum().item()
        metrics = Pack(num_samples=num_samples)
        target_len=target[1]
        mask=sequence_mask(target_len)
        mask=mask.float()
        # logits = outputs.logits
        # nll = self.nll_loss(logits, target)
        out_copy=outputs.out_copy
        #  out_copy  batch x max_len x src
        target_loss=out_copy.gather(2,target[0].unsqueeze(-1)).squeeze(-1)
        target_loss=target_loss*mask
        target_loss+=1e-15
        target_loss=target_loss.log()
        loss=-((target_loss.sum())/num_words)

        out_emo=outputs.logits    #  batch x  max_len x  dim
        batch_size, max_len, class_num=out_emo.size()
        # out_emo=out_emo.view(batch_size*max_len, class_num)
        # emo_target=emo_target.view(-1)
        target_emo_loss=out_emo.gather(2,emo_target[0].unsqueeze(-1)).squeeze(-1)
        target_len -= 1
        mask_ = sequence_mask(target_len)
        mask_ = mask_.float()
        new_mask=mask.data.new(batch_size,max_len).zero_()
        # print(mask.size())
        # print(new_mask.size())
        new_mask[:,:max_len-1]=mask_

        target_emo_loss = target_emo_loss * new_mask
        target_emo_loss += 1e-15
        target_emo_loss = target_emo_loss.log()
        emo_loss = -((target_emo_loss.sum()) / num_words)



        metrics.add(loss=loss)
        metrics.add(emo_loss=emo_loss)
        #  这里，我们将只计算
        acc = accuracy(out_copy, target[0], mask=mask)
        metrics.add(acc=acc)
        return metrics

    def iterate(self, inputs, optimizer=None, grad_clip=None, is_training=True, epoch=-1):
        """
        iterate
        """
        enc_inputs = inputs
        dec_inputs = inputs.num_tgt_input
        target = inputs.tgt_output
        emo_target= inputs.tgt_emo

        outputs = self.forward(enc_inputs, dec_inputs)
        metrics = self.collect_metrics(outputs, target, emo_target)

        loss = metrics.loss
        if epoch>self.pretrain_epoch:
            loss+=metrics.emo_loss
        if torch.isnan(loss):
            raise ValueError("nan loss encountered")

        if is_training:
            assert optimizer is not None
            optimizer.zero_grad()
            loss.backward()
            if grad_clip is not None and grad_clip > 0:
                clip_grad_norm_(parameters=self.parameters(),
                                max_norm=grad_clip)
            optimizer.step()
        return metrics
