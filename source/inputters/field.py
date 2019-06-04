#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/inputters/field.py
"""

import re
import nltk
import torch
from tqdm import tqdm
from collections import Counter

PAD = "<pad>"
UNK = "<unk>"
BOS = "<bos>"
EOS = "<end>"
NUM = "<num>"
ENTITY="<entity>"

def tokenize(s):
    """
    tokenize
    """
    s = re.sub('\d+', NUM, s).lower()
    # tokens = nltk.RegexpTokenizer(r'\w+|<sil>|[^\w\s]+').tokenize(s)
    tokens = s.split(' ')
    return tokens


class Field(object):
    """
    Field
    """
    def __init__(self,
                 sequential=False,
                 dtype=None):
        self.sequential = sequential
        self.dtype = dtype if dtype is not None else int

    def str2num(self, string):
        """
        str2num
        """
        raise NotImplementedError

    def num2str(self, number):
        """
        num2str
        """
        raise NotImplementedError

    def numericalize(self, strings):
        """
        numericalize
        """
        if isinstance(strings, str):
            return self.str2num(strings)
        else:
            return [self.numericalize(s) for s in strings]

    def denumericalize(self, numbers):
        """
        denumericalize
        """
        if isinstance(numbers, torch.Tensor):
            with torch.cuda.device_of(numbers):
                numbers = numbers.tolist()
        if self.sequential:
            if not isinstance(numbers[0], list):
                return self.num2str(numbers)
            else:
                return [self.denumericalize(x) for x in numbers]
        else:
            if not isinstance(numbers, list):
                return self.num2str(numbers)
            else:
                return [self.denumericalize(x) for x in numbers]


class NumberField(Field):
    """
    NumberField
    """
    def __init__(self,
                 sequential=False,
                 dtype=None):
        super(NumberField, self).__init__(sequential=sequential,
                                          dtype=dtype)

    def str2num(self, string):
        """
        str2num
        """
        if self.sequential:
            return [self.dtype(s) for s in string.split(" ")]
        else:
            return self.dtype(string)

    def num2str(self, number):
        """
        num2str
        """
        if self.sequential:
            return " ".join([str(x) for x in number])
        else:
            return str(number)


class TextField(Field):
    """
    TextField
    """
    def __init__(self,
                 pad_token=PAD,
                 unk_token=UNK,
                 bos_token=BOS,
                 eos_token=EOS,
                 entiy_token=ENTITY,
                 special_tokens=None,
                 embed_file=None,
                 entiy_dict_file=None,):
        super(TextField, self).__init__(sequential=True,
                                        dtype=int)

        self.pad_token = pad_token
        self.unk_token = unk_token
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.entiy_token=entiy_token
        self.embed_file = embed_file

        specials = [self.pad_token, self.unk_token,self.entiy_token,
                    self.bos_token, self.eos_token, ]
        self.specials = [x for x in specials if x is not None]

        if special_tokens is not None:
            for token in special_tokens:
                if token not in self.specials:
                    self.specials.append(token)
        self.entiy_dict = set()
        if entiy_dict_file:
            with open(entiy_dict_file, 'r', encoding='utf-8') as rf:
                for line in rf:
                    self.entiy_dict.add(line.strip())
            print('加载实体成功，size= %d'%(len(self.entiy_dict)))
        self.itoemo=['NORM', 'POS', 'NEG']
        self.emotoi={}
        for i, emo in enumerate(self.itoemo):
            self.emotoi[emo]=i

        self.itos = []
        self.stoi = {}
        self.vocab_size = 0
        self.embeddings = None

    def build_vocab(self, texts, min_freq=0, max_size=None):
        """
        build_vocab
        """

        counter = Counter()
        for text in tqdm(texts):
            # print(text.keys())
            counter.update(text)

        # frequencies of special tokens are not counted when building vocabulary
        # in frequency order
        for tok in self.specials:
            del counter[tok]

        self.itos = list(self.specials)

        if max_size is not None:
            max_size = max_size + len(self.itos)

        # sort by frequency, then alphabetically
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        cover = 0
        for word, freq in words_and_frequencies:
            if freq < min_freq or len(self.itos) == max_size:
                break
            self.itos.append(word)
            cover += freq
        cover = cover / sum(freq for _, freq in words_and_frequencies)
        print(
            "Built vocabulary of size {} (coverage: {:.3f})".format(len(self.itos), cover))

        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.vocab_size = len(self.itos)

        if self.embed_file is not None:
            self.embeddings = self.build_word_embeddings(self.embed_file)

    def build_word_embeddings(self, embed_file):
        """
        build_word_embeddings
        """
        if isinstance(embed_file, list):
            embeds = [self.build_word_embeddings(e_file)
                      for e_file in embed_file]
        elif isinstance(embed_file, dict):
            embeds = {e_name: self.build_word_embeddings(e_file)
                      for e_name, e_file in embed_file.items()}
        else:
            cover = 0
            print("Building word embeddings from '{}' ...".format(embed_file))
            with open(embed_file, "r") as f:
                num, dim = map(int, f.readline().strip().split())
                embeds = [[0] * dim] * len(self.stoi)
                for line in f:
                    w, vs = line.rstrip().split(maxsplit=1)
                    if w in self.stoi:
                        try:
                            vs = [float(x) for x in vs.split(" ")]
                        except Exception:
                            vs = []
                        if len(vs) == dim:
                            embeds[self.stoi[w]] = vs
                            cover += 1
            rate = cover / len(embeds)
            print("{} words have pretrained {}-D word embeddings (coverage: {:.3f})".format( \
                    cover, dim, rate))
        return embeds

    def dump_vocab(self):
        """
        dump_vocab
        """
        vocab = {"itos": self.itos,
                 "embeddings": self.embeddings}
        return vocab

    def load_vocab(self, vocab):
        """
        load_vocab
        """
        self.itos = vocab["itos"]
        self.stoi = {tok: i for i, tok in enumerate(self.itos)}
        self.vocab_size = len(self.itos)
        self.embeddings = vocab["embeddings"]

    def word2num(self, word):
        unk_idx = self.stoi[self.unk_token]
        entity_idx = self.stoi[self.entiy_token]
        if word in self.stoi:
            return self.stoi[word]
        elif word in self.entiy_dict:
            return entity_idx
        else:
            return unk_idx

    def easy_str2num(self, string):
        unk_idx = self.stoi[self.unk_token]
        indices = []
        for tok in string:
            if tok in self.stoi:
                indices.append(self.stoi[tok])
            else:
                indices.append(unk_idx)
        return indices

    def str2num(self, string):
        """
        str2num
        """
        unk_idx = self.stoi[self.unk_token]
        entity_idx= self.stoi[self.entiy_token]
        indices=[]
        for tok in string:
            if tok  in self.stoi:
                indices.append(self.stoi[tok])
            else:
                if tok in self.entiy_dict:
                    indices.append(entity_idx)
                else:
                    indices.append(unk_idx)



        # indices = [self.stoi.get(tok, unk_idx) for tok in tokens]
        return indices

    def num2str(self, number):
        """
        num2str
        """
        tokens = [self.itos[x] for x in number]
        if tokens[0] == self.bos_token:
            tokens = tokens[1:]
        text = []
        for w in tokens:
            if w != self.eos_token:
                text.append(w)
            else:
                break
        text = [w for w in text if w not in (self.pad_token, )]
        text = " ".join(text)
        return text

