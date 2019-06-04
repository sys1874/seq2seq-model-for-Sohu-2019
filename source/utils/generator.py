#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/utils/generator.py
"""

import torch

from source.utils.misc import sequence_mask
from source.utils.misc import list2tensor
from source.utils.misc import Pack


class TopKGenerator(object):
    """
    TopKGenerator
    """
    def __init__(self,
                 model,
                 src_field,
                 max_length=5,
                 ignore_unk=False,
                 length_average=True,
                 use_gpu=False,
                 beam_size=1):
        self.model = model.cuda() if use_gpu else model
        self.SRC = src_field
        self.k = beam_size
        self.max_length = max_length
        self.ignore_unk = ignore_unk
        self.length_average = length_average
        self.use_gpu = use_gpu
        self.PAD = src_field.stoi[src_field.pad_token]
        self.UNK = src_field.stoi[src_field.unk_token]
        self.BOS = src_field.stoi[src_field.bos_token]
        self.EOS = src_field.stoi[src_field.eos_token]
        self.V = self.SRC.vocab_size

    def forward(self, inputs, enc_hidden=None):
        """
        forward
        """
        # switch the model to evaluate mode
        self.model.eval()

        # lengths =inputs.num_src[1]
        # mask=sequence_mask(lengths).long()

        with torch.no_grad():
            enc_outputs, dec_state = self.model.encode(inputs, enc_hidden)
            # dec_state.mask=mask
            preds, lens, scores, emos = self.decode(dec_state, inputs)

        return enc_outputs, preds, lens, scores, emos

    def decode(self, dec_state, batch):
        """
        decode
        """
        long_tensor_type = torch.cuda.LongTensor if self.use_gpu else torch.LongTensor

        b = dec_state.get_batch_size()

        # [[0], [k*1], [k*2], ..., [k*(b-1)]]
        self.pos_index = (long_tensor_type(range(b)) * self.k).view(-1, 1)

        # Inflate the initial hidden states to be of size: (b*k, H)
        dec_state = dec_state.inflate(self.k)
        # print(len(dec_state.states))
        # Initialize the scores; for the first step,
        # ignore the inflated copies to avoid duplicate entries in the top k
        sequence_scores = long_tensor_type(b * self.k).float()
        sequence_scores.fill_(-float('inf'))
        sequence_scores.index_fill_(0, long_tensor_type(
            [i * self.k for i in range(b)]), 0.0)

        # Initialize the input vector
        input_var = long_tensor_type([self.BOS] * b * self.k)

        # Store decisions for backtracking
        stored_scores = list()
        stored_predecessors = list()
        stored_emitted_symbols = list()
        # stored_weizhi=list()
        stored_finished_sen = [[] for _ in range(b)]
        stored_word=[]
        stored_emotion=[]

        for t in range(1, self.max_length+1):
            # Run the RNN one step forward
            output, dec_state, attn = self.model.decode(input_var, dec_state)

            log_softmax_output = output.squeeze(1)
            emo_score, preds = log_softmax_output.max(-1)
            stored_emotion.append(preds.tolist())

            attn=attn.attn.squeeze(1)
            # attn=attn+emo_sc

            # attn=attn.log()
            # batch x max_len
            batch_size, max_len=attn.size()

            # To get the full sequence scores for the new candidates, add the
            # local scores for t_i to the predecessor scores for t_(i-1)
            sequence_scores = sequence_scores.unsqueeze(1).repeat(1, max_len)
            if self.length_average and t > 1:
                sequence_scores = sequence_scores * \
                    (1 - 1/t) + attn / t
            else:
                sequence_scores += attn

            scores, candidates = sequence_scores.view(
                b, -1).topk(self.k, dim=1)

            # Reshape input = (b*k, 1) and sequence_scores = (b*k)
            input_var = (candidates % max_len)

            # 对相关的位置
            sequence_scores = scores.view(b * self.k)

            input_var = input_var.view(b * self.k)

            # Update fields for next timestep
            predecessors = (
                candidates / max_len + self.pos_index.expand_as(candidates)).view(b * self.k)

            dec_state = dec_state.index_select(predecessors)
            # print(len(dec_state.states))
            # Update sequence scores and erase scores for end-of-sentence symbol so that they aren't expanded
            stored_scores.append(sequence_scores.clone())

            # 需要吧所有的 input_var 替换为@
            word_temp=[]
            mask_temp=[]

            # input_var_temp=[]
            for b_id in range(b):
                for k_id in range(self.k):
                    # print(b_id*self.k+k_id)
                    index=input_var[b_id*self.k+k_id].item()
                    if index> len(batch.raw_src[b_id])-1:
                        word = batch.raw_src[b_id][len(batch.raw_src[b_id])-1]
                    else:
                        word=batch.raw_src[b_id][index]
                    # src_index=batch.num_src[0][b_id][input_var[b_id*self.k+k_id].item()].item()
                    # print(src_index.size())
                    # import  sys
                    # sys.exit()
                    # input_var_temp.append(src_index)
                    word_temp.append(word)
                    mask=dec_state.mask[b_id*self.k+k_id].tolist()

                    # print('修改前的mask值 ：%d'%(sum(mask)))
                    assert len(batch.raw_src[b_id]) <= len(mask)
                    for i ,w in enumerate(batch.raw_src[b_id]):
                        if w==word:
                            # print('发现一样的单词')
                            # print(mask[i])
                            mask[i]=0  #  屏蔽掉已经找到的单词
                    # print('修改后的mask值 ：%d' % (sum(mask)))
                    mask_temp.append(mask)

            stored_word.append(word_temp)
            # weizhi_index=input_var
            # stored_weizhi.append(weizhi_index)
            input_var=long_tensor_type([self.SRC.word2num(x) for x in word_temp])
            # input_var=long_tensor_type(input_var_temp)
            dec_state.mask=long_tensor_type(mask_temp)

            eos_indices = input_var.data.eq(self.EOS)

            eos_indices_1 = eos_indices.nonzero()

            if eos_indices_1.dim() > 0:
                # print('有查找到节点')
                # if t>=3:
                for i in range(0, eos_indices_1.size(0)):
                            # Indices of the EOS symbol for both variables
                            # with b*k as the first dimension, and b, k for
                            # the first two dimensions
                            idx = eos_indices_1[i]
                            b_idx = idx[0].item() // self.k
                            stored_finished_sen[b_idx].append((sequence_scores[idx[0]].item(), t - 1, idx[0].item()))

                sequence_scores.data.masked_fill_(eos_indices, -float('inf'))
                # 让 end 对应的 句子，停止采样

            # eos_indices = input_var.data.eq(self.EOS)
            # if eos_indices.nonzero().dim() > 0:
            #     sequence_scores.data.masked_fill_(eos_indices, -float('inf'))

            # if self.ignore_unk:
            #     # Erase scores for UNK symbol so that they aren't expanded
            #     unk_indices = input_var.data.eq(self.UNK)
            #     if unk_indices.nonzero().dim() > 0:
            #         sequence_scores.data.masked_fill_(
            #             unk_indices, -float('inf'))

            # Cache results for backtracking
            stored_predecessors.append(predecessors)
            stored_emitted_symbols.append(input_var)
        predicts = []
        lengths = []
        scores = []
        emos=[]
        for b_id in range(b):
            i = 0
            while len(stored_finished_sen[b_id]) < 1:
                score = stored_scores[-1][b_id * self.k + i].item()
                stored_finished_sen[b_id].append((score, t - 1, b_id * self.k + i))
                i += 1
            stored_finished_sen[b_id].sort(key=lambda x: -x[0])
            scores.append([stored_finished_sen[b_id][0][0]])
            pre, emo = self._get_hyp(stored_emotion, stored_predecessors, stored_word, stored_finished_sen[b_id][0][2], \
                                stored_finished_sen[b_id][0][1])
            predicts.append([pre])
            emos.append([emo])
            lengths.append(len(pre))
        # predicts, scores, lengths = self._backtrack(
        #     stored_weizhi, stored_predecessors, stored_emitted_symbols, stored_scores, b)

        # predicts = predicts[:, :1]
        # scores = scores[:, :1]
        # lengths = long_tensor_type(lengths)[:, :1]
        # mask = sequence_mask(lengths, max_len=self.max_length).eq(0)
        # predicts[mask] = self.PAD
        #  [[[],[],],[[],[],]]
        return predicts, lengths, scores, emos

    def _get_hyp(self, stored_emotion, stored_predecessors, stored_emitted_symbols, b_id, t):
        # stored_predecessors：  [[上一个state的 位置]]
        # stored_emitted_symbols：  [[这次预测的位置]]
        # stored_scores  [[这次预测的得分]]      b*k
        pre=[]
        emo=[]
        for j in range(t,-1,-1):
            emo.append(self.SRC.itoemo[stored_emotion[j][b_id]])
            pre.append(stored_emitted_symbols[j][b_id])
            b_id=stored_predecessors[j][b_id]
        if pre[0]==self.SRC.eos_token:
            pre=pre[1:]
            emo=emo[1:]
        return pre[::-1], emo[::-1]

    def _backtrack(self, stored_weizhi, predecessors, symbols, scores, b):
        p = list()
        l = [[self.max_length] * self.k for _ in range(b)]

        # the last step output of the beams are not sorted
        # thus they are sorted here
        sorted_score, sorted_idx = scores[-1].view(
            b, self.k).topk(self.k, dim=1)

        # initialize the sequence scores with the sorted last step beam scores
        s = sorted_score.clone()

        # the number of EOS found in the backward loop below for each batch
        batch_eos_found = [0] * b

        t = self.max_length - 1
        # initialize the back pointer with the sorted order of the last step beams.
        # add self.pos_index for indexing variable with b*k as the first dimension.
        t_predecessors = (
            sorted_idx + self.pos_index.expand_as(sorted_idx)).view(b * self.k)

        while t >= 0:
            # Re-order the variables with the back pointer
            current_symbol = symbols[t].index_select(0, t_predecessors)
            # Re-order the back pointer of the previous step with the back pointer of the current step
            t_predecessors = predecessors[t].index_select(0, t_predecessors)

            # This tricky block handles dropped sequences that see EOS earlier.
            # The basic idea is summarized below:
            #
            #   Terms:
            #       Ended sequences = sequences that see EOS early and dropped
            #       Survived sequences = sequences in the last step of the beams
            #
            #       Although the ended sequences are dropped during decoding,
            #   their generated symbols and complete backtracking information are still
            #   in the backtracking variables.
            #   For each batch, everytime we see an EOS in the backtracking process,
            #       1. If there is survived sequences in the return variables, replace
            #       the one with the lowest survived sequence score with the new ended
            #       sequences
            #       2. Otherwise, replace the ended sequence with the lowest sequence
            #       score with the new ended sequence
            #
            eos_indices = symbols[t].data.eq(self.EOS).nonzero()
            #
            if eos_indices.dim() > 0:
                for i in range(eos_indices.size(0)-1, -1, -1):
                    # Indices of the EOS symbol for both variables
                    # with b*k as the first dimension, and b, k for
                    # the first two dimensions
                    idx = eos_indices[i]
                    b_idx = idx[0].item() // self.k
                    # The indices of the replacing position
                    # according to the replacement strategy noted above
                    res_k_idx = self.k - (batch_eos_found[b_idx] % self.k) - 1
                    batch_eos_found[b_idx] += 1
                    res_idx = b_idx * self.k + res_k_idx

                    # Replace the old information in return variables
                    # with the new ended sequence information
                    t_predecessors[res_idx] = predecessors[t][idx[0]]
                    current_symbol[res_idx] = symbols[t][idx[0]]
                    s[b_idx, res_k_idx] = scores[t][idx[0]]
                    l[b_idx][res_k_idx] = t + 1

            # record the back tracked results
            p.append(current_symbol)

            t -= 1

        # Sort and re-order again as the added ended sequences may change
        # the order (very unlikely)
        s, re_sorted_idx = s.topk(self.k)
        for b_idx in range(b):
            l[b_idx] = [l[b_idx][k_idx.item()]
                        for k_idx in re_sorted_idx[b_idx, :]]

        re_sorted_idx = (
            re_sorted_idx + self.pos_index.expand_as(re_sorted_idx)).view(b * self.k)

        # Reverse the sequences and re-order at the same time
        # It is reversed because the backtracking happens in reverse time order
        predicts = torch.stack(p[::-1]).t()
        predicts = predicts[re_sorted_idx].contiguous().view(
            b, self.k, -1).data
        # p = [step.index_select(0, re_sorted_idx).view(b, self.k).data for step in reversed(p)]
        scores = s.data
        lengths = l

        # if self.k == 1:
        #     lengths = [_l[0] for _l in lengths]

        return predicts, scores, lengths

    def generate(self, batch_iter, num_batches=None):
        """
        generate
        """
        results = []
        batch_cnt = 0
        for batch in batch_iter:
            enc_outputs, preds, lengths, scores, emos = self.forward(
                inputs=batch, enc_hidden=None)

            # denumericalization
            # src = batch.src[0]
            # tgt = batch.tgt[0]
            # src = self.src_field.denumericalize(src)
            # tgt = self.tgt_field.denumericalize(tgt)
            # preds = self.tgt_field.denumericalize(preds)
            # scores = scores.tolist()
            temp=[]
            tgt_emo=batch.tgt_emo[0].tolist()
            for item in tgt_emo:
                temp.append([self.SRC.itoemo[x] for x in item])



            # if 'cue' in batch:
            #     cue = self.tgt_field.denumericalize(batch.cue[0].data)
            #     enc_outputs.add(cue=cue)
            if hasattr(batch, 'raw_tgt') and batch.raw_tgt is not None:
                tgt=batch.raw_tgt
                enc_outputs.add(tgt=tgt, preds=preds, scores=scores, emos=emos, target_emos=temp)
            else:
                enc_outputs.add(preds=preds, scores=scores, emos=emos, target_emos=temp)
            if hasattr(batch, 'id') and batch.id is not None:
                enc_outputs.add(id=batch['id'])
            result_batch = enc_outputs.flatten()
            results += result_batch
            batch_cnt += 1
            if batch_cnt == num_batches:
                break
        return results

    def interact(self, src, cue=None):
        """
        interact
        """
        if src == "":
            return None

        inputs = Pack()
        src = self.src_field.numericalize([src])
        inputs.add(src=list2tensor(src))

        if cue is not None:
            cue = self.cue_field.numericalize([cue])
            inputs.add(cue=list2tensor(cue))
        if self.use_gpu:
            inputs = inputs.cuda()
        _, preds, _, _ = self.forward(inputs=inputs)

        pred = self.tgt_field.denumericalize(preds[0][0])

        return pred

