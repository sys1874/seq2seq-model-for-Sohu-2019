#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/encoders/attention.py
"""

import torch
import torch.nn as nn
import math
# from source.utils.misc import sequence_mask
# from source.utils.misc import cnn_self_attn_mask

class Attention(nn.Module):
    """
    Attention
    """
    def __init__(self,
                 query_size,
                 memory_size=None,
                 hidden_size=None,
                 mode="mlp",
                 return_attn_only=False,
                 project=False):
        super(Attention, self).__init__()
        assert (mode in ["dot", "general", "mlp", "m_head", "csan"]), (
            "Unsupported attention mode: {mode}"
        )

        self.query_size = query_size
        self.memory_size = memory_size or query_size
        self.hidden_size = hidden_size or query_size
        self.mode = mode
        self.return_attn_only = return_attn_only
        self.project = project

        if mode == "general":
            self.linear_query = nn.Linear(
                self.query_size, self.memory_size, bias=False)
        elif mode == "mlp":
            self.linear_query = nn.Linear(
                self.query_size, self.hidden_size, bias=True)
            self.linear_memory = nn.Linear(
                self.memory_size, self.hidden_size, bias=False)
            self.tanh = nn.Tanh()
            self.v = nn.Linear(self.hidden_size, 1, bias=False)
        elif mode=='m_head' or mode=="csan":
            self.head_count = 8
            self.dim_per_head= query_size // self.head_count
            assert (self.dim_per_head*self.head_count)==query_size
            self.linear_keys = nn.Linear(query_size,
                                         self.head_count * self.dim_per_head)
            self.linear_values = nn.Linear(query_size,
                                           self.head_count * self.dim_per_head)
            self.linear_query = nn.Linear(query_size,
                                          self.head_count * self.dim_per_head)
            self.dropout = nn.Dropout(0.1)
            self.final_linear = nn.Linear(query_size, query_size)
        self.softmax = nn.Softmax(dim=-1)

        if self.project:
            self.linear_project = nn.Sequential(
                nn.Linear(in_features=self.hidden_size + self.memory_size,
                          out_features=self.hidden_size),
                nn.Tanh())

    def __repr__(self):
        main_string = "Attention({}, {}".format(self.query_size, self.memory_size)
        if self.mode == "mlp":
            main_string += ", {}".format(self.hidden_size)
        main_string += ", mode='{}'".format(self.mode)
        if self.project:
            main_string += ", project=True"
        main_string += ")"
        return main_string

    def forward(self, query, memory, mask=None):
        """
        query: Tensor(batch_size, query_length, query_size)
        memory: Tensor(batch_size, memory_length, memory_size)
        mask: Tensor(batch_size, memory_length)
        """
        if self.mode == "dot":
            assert query.size(-1) == memory.size(-1)
            # (batch_size, query_length, memory_length)
            attn = torch.bmm(query, memory.transpose(1, 2))
        elif self.mode == "general":
            assert self.memory_size == memory.size(-1)
            # (batch_size, query_length, memory_size)
            key = self.linear_query(query)
            # (batch_size, query_length, memory_length)
            attn = torch.bmm(key, memory.transpose(1, 2))
        elif self.mode == "mlp":
            # (batch_size, query_length, memory_length, hidden_size)
            hidden = self.linear_query(query).unsqueeze(
                2) + self.linear_memory(memory).unsqueeze(1)
            key = self.tanh(hidden)
            # (batch_size, query_length, memory_length)
            attn = self.v(key).squeeze(-1)
        elif self.mode=="m_head" or self.mode=="csan":
            batch_size=query.size(0)
            head_count=self.head_count
            dim_per_head=self.dim_per_head
            def shape(x):
                """  projection """
                return x.view(batch_size, -1, head_count, dim_per_head) \
                    .transpose(1, 2)

            def unshape(x):
                """  compute context """
                return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_count * dim_per_head)

            key = self.linear_keys(memory)
            value = self.linear_values(memory)
            query = self.linear_query(query)
            key = shape(key)
            value = shape(value)
            query = shape(query)
            query = query / math.sqrt(dim_per_head)
            attn = torch.matmul(query, key.transpose(2, 3))  # batch x head x  q_l x m_l

        if mask is not None:
            # (batch_size, query_length, memory_length)
            if self.mode=="m_head":
                mask = mask.unsqueeze(1).repeat(1, head_count,  1)
                mask = mask.unsqueeze(2).repeat(1, 1, query.size(2), 1)
                # print(attn.size())
                # print(mask.size())
                attn.masked_fill_(mask, -float("inf"))
            elif self.mode=="csan":
                mask = mask.unsqueeze(1).repeat(1, head_count, 1, 1)
                attn.masked_fill_(mask, -1e20)
            else:
                mask = mask.unsqueeze(1).repeat(1, query.size(1), 1)
                # attn.masked_fill_(mask, -float("inf"))
                attn.masked_fill_(mask, -1e20)

        # (batch_size, query_length, memory_length)
        weights = self.softmax(attn)
        if self.return_attn_only:
            return weights

        # (batch_size, query_length, memory_size)
        if self.mode=='m_head' or self.mode=='csan':
            drop_attn = self.dropout(weights)
            context = unshape(torch.matmul(drop_attn, value))

            weighted_memory = self.final_linear(context)
            weights = weights[:, 0, :, :]
        else:
            weighted_memory = torch.bmm(weights, memory)

        if self.project:
            project_output = self.linear_project(
                torch.cat([weighted_memory, query], dim=-1))
            return project_output, weights
        else:
            return weighted_memory, weights
