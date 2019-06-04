#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/inputters/corpus.py
"""

import os
import torch

from tqdm import tqdm
from source.inputters.field import TextField
from source.inputters.dataset import Dataset, Entity_Dataset, Entity_Dataset_pos

import json

class Corpus(object):
    """
    Corpus
    """
    def __init__(self,
                 data_dir,
                 data_prefix,
                 min_freq=0,
                 max_vocab_size=None):
        self.data_dir = data_dir
        self.data_prefix = data_prefix
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size

        prepared_data_file = data_prefix + "_" + str(max_vocab_size) + ".data.pt"
        prepared_vocab_file = data_prefix + "_" + str(max_vocab_size) + ".vocab.pt"

        self.prepared_data_file = os.path.join(data_dir, prepared_data_file)
        self.prepared_vocab_file = os.path.join(data_dir, prepared_vocab_file)
        self.fields = {}
        self.filter_pred = None
        self.sort_fn = None
        self.data = None

    def load(self):
        """
        load
        """
        if not (os.path.exists(self.prepared_data_file) and
                os.path.exists(self.prepared_vocab_file)):
            self.build()
        self.load_vocab(self.prepared_vocab_file)
        self.load_data(self.prepared_data_file)

        self.padding_idx = self.TGT.stoi[self.TGT.pad_token]

    def reload(self, data_type='test'):
        """
        reload
        """
        data_file = os.path.join(self.data_dir, self.data_prefix + "." + data_type)
        data_raw = self.read_data(data_file, data_type="test")
        data_examples = self.build_examples(data_raw)
        self.data[data_type] = Dataset(data_examples)

        print("Number of examples:",
              " ".join("{}-{}".format(k.upper(), len(v)) for k, v in self.data.items()))

    def load_data(self, prepared_data_file=None):
        """
        load_data
        """
        prepared_data_file = prepared_data_file or self.prepared_data_file
        print("Loading prepared data from {} ...".format(prepared_data_file))
        data = torch.load(prepared_data_file)
        self.data = {"train": Dataset(data['train']),
                     "valid": Dataset(data["valid"]),
                     "test": Dataset(data["test"])}
        print("Number of examples:",
              " ".join("{}-{}".format(k.upper(), len(v)) for k, v in self.data.items()))

    def load_vocab(self, prepared_vocab_file):
        """
        load_vocab
        """
        prepared_vocab_file = prepared_vocab_file or self.prepared_vocab_file
        print("Loading prepared vocab from {} ...".format(prepared_vocab_file))
        vocab_dict = torch.load(prepared_vocab_file)

        for name, vocab in vocab_dict.items():
            if name in self.fields:
                self.fields[name].load_vocab(vocab)
        print("Vocabulary size of fields:",
              " ".join("{}-{}".format(name.upper(), field.vocab_size) 
                for name, field in self.fields.items() 
                    if isinstance(field, TextField)))

    def read_data(self, data_file, data_type=None):
        """
        Returns
        -------
        data: ``List[Dict]``
        """
        raise NotImplementedError

    def build_vocab(self, data):
        """
        Args
        ----
        data: ``List[Dict]``
        """
        field_data_dict = {}
        for name in data[0].keys():
            field = self.fields.get(name)
            if isinstance(field, TextField):
                xs = [x[name] for x in data]
                if field not in field_data_dict:
                    field_data_dict[field] = xs
                else:
                    field_data_dict[field] += xs

        vocab_dict = {}
        for name, field in self.fields.items():
            if field in field_data_dict:
                print("Building vocabulary of field {} ...".format(name.upper()))
                if field.vocab_size == 0:
                    field.build_vocab(field_data_dict[field],
                                      min_freq=self.min_freq,
                                      max_size=self.max_vocab_size)
                vocab_dict[name] = field.dump_vocab()
        return vocab_dict

    def build_examples(self, data):
        """
        Args
        ----
        data: ``List[Dict]``
        """
        examples = []
        for raw_data in tqdm(data):
            example = {}
            for name, strings in raw_data.items():
                example[name] = self.fields[name].numericalize(strings)
            examples.append(example)
        if self.sort_fn is not None:
            print("Sorting examples ...")
            examples = self.sort_fn(examples)
        return examples

    def build(self):
        """
        build
        """
        print("Start to build corpus!")
        train_file = os.path.join(self.data_dir, self.data_prefix + ".train")


        print("Reading data ...")
        train_raw = self.read_data(train_file, data_type="train")

        vocab = self.build_vocab(train_raw)

        print("Building TRAIN examples ...")
        train_data = self.build_examples(train_raw)
        print("Building VALID examples ...")
        # valid_data = self.build_examples(valid_raw)
        # print("Building TEST examples ...")
        # test_data = self.build_examples(test_raw)

        # data = {"train": train_data,
        #         "valid": valid_data,
        #         "test": test_data}

        print("Saving prepared vocab ...")
        torch.save(vocab, self.prepared_vocab_file)
        print("Saved prepared vocab to '{}'".format(self.prepared_vocab_file))

        print("Saving prepared data ...")
        torch.save(train_data, self.prepared_data_file)
        print("Saved prepared data to '{}'".format(self.prepared_data_file))

    def create_batches(self, batch_size, data_type="train",
                       shuffle=False, device=None):
        """
        create_batches
        """
        try:
            data = self.data[data_type]
            data_loader = data.create_batches(batch_size, shuffle, device)
            return data_loader
        except KeyError:
            raise KeyError("Unsported data type: {}!".format(data_type))

    def transform(self, data_file, batch_size,
                  data_type="test", shuffle=False, device=None):
        """
        Transform raw text from data_file to Dataset and create data loader.
        """
        raw_data = self.read_data(data_file, data_type=data_type)
        examples = self.build_examples(raw_data)
        data = Dataset(examples)
        data_loader = data.create_batches(batch_size, shuffle, device)
        return data_loader



class Entity_Corpus(object):
    """
    Corpus
    """
    def __init__(self,
                 data_dir,
                 data_prefix,
                 min_freq=0,
                 entity_file=None,
                 max_vocab_size=None):
        self.data_dir = data_dir
        self.data_prefix = data_prefix
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size

        prepared_data_file = data_prefix + "_" + str(max_vocab_size) + ".data.pt"
        prepared_vocab_file = data_prefix + "_" + str(max_vocab_size) + ".vocab.pt"

        self.prepared_data_file = os.path.join(data_dir, prepared_data_file)
        self.prepared_vocab_file = os.path.join(data_dir, prepared_vocab_file)
        self.SRC = TextField(entiy_dict_file=entity_file)
        self.filter_pred = None
        self.sort_fn = None
        self.data = None

    def load(self):
        """
        load
        """
        if not (os.path.exists(self.prepared_data_file) and
                os.path.exists(self.prepared_vocab_file)):
            self.build()
        self.load_vocab(self.prepared_vocab_file)
        self.load_data(self.prepared_data_file)

        self.padding_idx = self.SRC.stoi[self.SRC.pad_token]

    def reload(self, data_type='test'):
        """
        reload
        """
        data_file = os.path.join(self.data_dir, self.data_prefix + "." + data_type)
        if os.path.exists(data_file):
            self.data[data_type] = Entity_Dataset(torch.load(data_file))
        else:
            data_file_raw=data_file+'.raw'
            data_raw = self.read_data(data_file_raw, data_type="test", has_id=True)
            data_examples = self.build_examples(data_raw)
            torch.save(data_examples, data_file)
            self.data[data_type] = Entity_Dataset(data_examples)

        print("Number of examples:",
              " ".join("{}-{}".format(k.upper(), len(v)) for k, v in self.data.items()))

    def load_data(self, prepared_data_file=None):
        """
        load_data
        """
        prepared_data_file = prepared_data_file or self.prepared_data_file
        print("Loading prepared data from {} ...".format(prepared_data_file))
        data = torch.load(prepared_data_file)
        self.data = {"train": Entity_Dataset(data['train']),
                     "valid": Entity_Dataset(data["valid"]),
                    }
        print("Number of examples:",
              " ".join("{}-{}".format(k.upper(), len(v)) for k, v in self.data.items()))

    def load_vocab(self, prepared_vocab_file):
        """
        load_vocab
        """
        prepared_vocab_file = prepared_vocab_file or self.prepared_vocab_file
        print("Loading prepared vocab from {} ...".format(prepared_vocab_file))
        vocab_dict = torch.load(prepared_vocab_file)
        self.SRC.load_vocab(vocab_dict)
        print('Finish loading vocab , size: %d'%(self.SRC.vocab_size))


    def read_data(self, data_file, data_type="train", has_id=False):
        """
        read_data
        """
        data = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                line = json.loads(line)
                src = line['text']
                mask = line['mask']
                true_Entity = line['true_Entity']


                # 建立 tgt
                tgt = []
                # 添加一个开始表示符
                pre = '<bos>'
                for item in true_Entity:
                    tgt.append([pre, item[0], item[1]['emotion']])
                    pre = item[1]['entity']
                tgt.append([pre, len(src) - 1, 'NORM'])
                #  指向最后一位
                d = {'src': src, 'mask': mask, 'tgt': tgt}
                if has_id:
                    d['id']=line['newsId']
                data.append(d)
        # 划分一下 训练验证
        print('finished data read')
        return data

    def build_vocab(self, data):
        """
        Args
        ----
        data: ``List[Dict]``
        """
        src = [x['src'] for x in data]
        self.SRC.build_vocab(src, min_freq=self.min_freq,
                            max_size=self.max_vocab_size)
        return  self.SRC.dump_vocab()

    def build_examples(self, data):
        """
        Args
        ----
        data: ``List[Dict]``

        raw_data  :
        text
        mask
        tgt

        """
        examples = []
        for raw_data in tqdm(data):
            example = {}
            raw_text=raw_data['src']
            num_text=self.SRC.str2num(raw_text)
            example['num_src']=num_text
            example['raw_src']=raw_text
            example['mask']=raw_data['mask']

            tgt_input=[]
            tgt_output=[]
            raw_tgt=[]
            tgt_emo=[]


            for [input,output,emotion] in raw_data['tgt']:
                raw_tgt.append(input)
                tgt_input.append(self.SRC.word2num(input))
                tgt_emo.append(self.SRC.emotoi.get(emotion, 0))
                tgt_output.append(output)


            example['num_tgt_input']=tgt_input
            example['tgt_output']=tgt_output
            example['tgt_emo']=tgt_emo
            example['raw_tgt']=raw_tgt
            if 'id' in raw_data:
                example['id']=raw_data['id']
            examples.append(example)
        if self.sort_fn is not None:
            print("Sorting examples ...")
            examples = self.sort_fn(examples)
        return examples

    def build(self):
        """
        build
        """
        print("Start to build corpus!")
        train_file = os.path.join(self.data_dir, self.data_prefix + ".train")


        print("Reading data ...")
        train_raw = self.read_data(train_file, data_type="train")

        vocab = self.build_vocab(train_raw)

        print("Building TRAIN examples ...")
        train_data = self.build_examples(train_raw)
        import  random
        random.shuffle(train_data)
        # train_data=train_data
        # valid_data=train_data
        valid_data=train_data[:2000]
        train_data=train_data[2000:]


        data = {"train": train_data,
                "valid": valid_data,
                }

        print('num_train_data %d, num_valid_data %d'%(len(train_data), len(valid_data)))

        print("Saving prepared vocab ...")
        torch.save(vocab, self.prepared_vocab_file)
        print("Saved prepared vocab to '{}'".format(self.prepared_vocab_file))

        print("Saving prepared data ...")
        torch.save(data, self.prepared_data_file)
        print("Saved prepared data to '{}'".format(self.prepared_data_file))

    def create_batches(self, batch_size, data_type="train",
                       shuffle=False, device=None):
        """
        create_batches
        """
        try:
            data = self.data[data_type]
            data_loader = data.create_batches(batch_size, shuffle, device)
            return data_loader
        except KeyError:
            raise KeyError("Unsported data type: {}!".format(data_type))

    def transform(self, data_file, batch_size,
                  data_type="test", shuffle=False, device=None):
        """
        Transform raw text from data_file to Dataset and create data loader.
        """
        raw_data = self.read_data(data_file, data_type=data_type)
        examples = self.build_examples(raw_data)
        data = Dataset(examples)
        data_loader = data.create_batches(batch_size, shuffle, device)
        return data_loader


class Entity_Corpus_pos(object):
    """
    Corpus
    """
    def __init__(self,
                 data_dir,
                 data_prefix,
                 min_freq=0,
                 entity_file=None,
                 max_vocab_size=None):
        self.data_dir = data_dir
        self.data_prefix = data_prefix
        self.min_freq = min_freq
        self.max_vocab_size = max_vocab_size

        prepared_data_file = data_prefix + "_" + str(max_vocab_size) + ".data.pt"
        prepared_vocab_file = data_prefix + "_" + str(max_vocab_size) + ".vocab.pt"

        self.prepared_data_file = os.path.join(data_dir, prepared_data_file)
        self.prepared_vocab_file = os.path.join(data_dir, prepared_vocab_file)
        self.SRC = TextField(entiy_dict_file=entity_file)
        self.POS = TextField(bos_token=None, eos_token=None)

        self.filter_pred = None
        self.sort_fn = None
        self.data = None

    def load(self):
        """
        load
        """
        if not (os.path.exists(self.prepared_data_file) and
                os.path.exists(self.prepared_vocab_file)):
            self.build()
        self.load_vocab(self.prepared_vocab_file)
        self.load_data(self.prepared_data_file)

        self.padding_idx = self.SRC.stoi[self.SRC.pad_token]

    def reload(self, data_type='test'):
        """
        reload
        """
        data_file = os.path.join(self.data_dir, self.data_prefix + "." + data_type)
        if os.path.exists(data_file):
            self.data[data_type] = Entity_Dataset_pos(torch.load(data_file))
        else:
            data_file_raw=data_file+'.raw'
            data_raw = self.read_data(data_file_raw, data_type="test", has_id=True)
            data_examples = self.build_examples(data_raw)
            torch.save(data_examples, data_file)
            self.data[data_type] = Entity_Dataset_pos(data_examples)

        print("Number of examples:",
              " ".join("{}-{}".format(k.upper(), len(v)) for k, v in self.data.items()))

    def load_data(self, prepared_data_file=None):
        """
        load_data
        """
        prepared_data_file = prepared_data_file or self.prepared_data_file
        print("Loading prepared data from {} ...".format(prepared_data_file))
        data = torch.load(prepared_data_file)
        self.data = {"train": Entity_Dataset_pos(data['train']),
                     "valid": Entity_Dataset_pos(data["valid"]),
                    }
        print("Number of examples:",
              " ".join("{}-{}".format(k.upper(), len(v)) for k, v in self.data.items()))

    def load_vocab(self, prepared_vocab_file):
        """
        load_vocab
        """
        prepared_vocab_file = prepared_vocab_file or self.prepared_vocab_file
        print("Loading prepared vocab from {} ...".format(prepared_vocab_file))
        vocab_dict = torch.load(prepared_vocab_file)
        self.SRC.load_vocab(vocab_dict['src'])
        self.POS.load_vocab(vocab_dict['pos'])
        print('Finish loading vocab , size: %d'%(self.SRC.vocab_size))
        print('Finish loading pos vocab , size: %d' % (self.POS.vocab_size))


    def read_data(self, data_file, data_type="train", has_id=False):
        """
        read_data
        """
        data = []
        with open(data_file, "r", encoding="utf-8") as f:
            import os
            LTP_DATA_DIR = './extend/ltp_data_v3.4.0'
            from pyltp import Postagger
            pos_model_path = os.path.join(LTP_DATA_DIR, 'pos.model')
            postagger = Postagger()  # 初始化实例
            postagger.load(pos_model_path)  # 加载模型
            for line in f:
                line = json.loads(line)
                src = line['text']
                mask = line['mask']
                true_Entity = line['true_Entity']

                pos = list(postagger.postag(src))

                assert  len(src) == len(pos)
                # 建立 tgt
                tgt = []
                # 添加一个开始表示符
                pre = '<bos>'
                for item in true_Entity:
                    tgt.append([pre, item[0], item[1]['emotion']])
                    pre = item[1]['entity']
                tgt.append([pre, len(src) - 1, 'NORM'])
                #  指向最后一位
                d = {'src': src, 'mask': mask, 'tgt': tgt, 'pos':pos}
                if has_id:
                    d['id']=line['newsId']
                data.append(d)
        # 划分一下 训练验证
        print('finished data read')
        return data

    def build_vocab(self, data):
        """
        Args
        ----
        data: ``List[Dict]``
        """

        temp1=[x['src'] for x in data]
        self.SRC.build_vocab(temp1, min_freq=self.min_freq,
                            max_size=self.max_vocab_size)
        temp2 = [x['pos'] for x in data]
        self.POS.build_vocab(temp2, min_freq=self.min_freq,
                            max_size=self.max_vocab_size)
        d={}
        d['src']=self.SRC.dump_vocab()
        d['pos']=self.POS.dump_vocab()

        return  d

    def build_examples(self, data):
        """
        Args
        ----
        data: ``List[Dict]``

        raw_data  :
        text
        mask
        tgt

        """
        examples = []
        for raw_data in tqdm(data):
            example = {}
            raw_text=raw_data['src']
            num_text=self.SRC.str2num(raw_text)

            example['num_pos']=self.POS.easy_str2num(raw_data['pos'])
            example['num_src']=num_text
            example['raw_src']=raw_text

            example['mask']=raw_data['mask']
            # raw_tgt,tgt_output,tgt_emo=zip(*raw_data['tgt'])

            # tgt_input = []
            # tgt_output=list(tgt_output)
            # tgt_input.append(self.SRC.word2num(raw_tgt[0]))
            # tgt_input+=[num_text[x] for x in tgt_output[:-1]]
            # tgt_emo=[self.SRC.emotoi.get(emotion, 0) for emotion in tgt_emo]
            # assert len(tgt_input) ==len(tgt_output)

            tgt_input=[]
            tgt_output=[]
            raw_tgt=[]
            tgt_emo=[]


            for [input,output,emotion] in raw_data['tgt']:
                raw_tgt.append(input)
                tgt_input.append(self.SRC.word2num(input))
                # tgt_input.append(self.SRC.target2num(input))
                tgt_emo.append(self.SRC.emotoi.get(emotion, 0))
                # if input in self.SRC.stoi:
                #     tgt_input.append(self.SRC.stoi[input])
                # elif input in self.SRC.entiy_dict:
                #     tgt_input.append(self.SRC.stoi[self.SRC.entiy_token])
                # else:
                #     tgt_input.append(self.SRC.stoi[self.SRC.unk_token])
                tgt_output.append(output)


            example['num_tgt_input']=tgt_input
            example['tgt_output']=tgt_output
            example['tgt_emo']=tgt_emo
            example['raw_tgt']=raw_tgt
            if 'id' in raw_data:
                example['id']=raw_data['id']
            examples.append(example)
        if self.sort_fn is not None:
            print("Sorting examples ...")
            examples = self.sort_fn(examples)
        return examples

    def build(self):
        """
        build
        """
        print("Start to build corpus!")
        train_file = os.path.join(self.data_dir, self.data_prefix + ".train")


        print("Reading data ...")
        train_raw = self.read_data(train_file, data_type="train")
        if  os.path.exists(self.prepared_vocab_file):
            print('加载旧字典')
            self.load_vocab(self.prepared_vocab_file)
        else:
            vocab = self.build_vocab(train_raw)
        # vocab = self.build_vocab(train_raw)

        print("Building TRAIN examples ...")
        train_data = self.build_examples(train_raw)
        import  random
        random.shuffle(train_data)
        # train_data=train_data
        # valid_data=train_data
        valid_data=train_data[:2000]
        train_data=train_data[2000:]


        data = {"train": train_data,
                "valid": valid_data,
                }

        print('num_train_data %d, num_valid_data %d'%(len(train_data), len(valid_data)))

        if not os.path.exists(self.prepared_vocab_file):
            print("Saving prepared vocab ...")
            torch.save(vocab, self.prepared_vocab_file)
            print("Saved prepared vocab to '{}'".format(self.prepared_vocab_file))
        # print("Saving prepared vocab ...")
        # torch.save(vocab, self.prepared_vocab_file)
        # print("Saved prepared vocab to '{}'".format(self.prepared_vocab_file))

        print("Saving prepared data ...")
        torch.save(data, self.prepared_data_file)
        print("Saved prepared data to '{}'".format(self.prepared_data_file))

    def create_batches(self, batch_size, data_type="train",
                       shuffle=False, device=None):
        """
        create_batches
        """
        try:
            data = self.data[data_type]
            data_loader = data.create_batches(batch_size, shuffle, device)
            return data_loader
        except KeyError:
            raise KeyError("Unsported data type: {}!".format(data_type))

    def transform(self, data_file, batch_size,
                  data_type="test", shuffle=False, device=None):
        """
        Transform raw text from data_file to Dataset and create data loader.
        """
        raw_data = self.read_data(data_file, data_type=data_type)
        examples = self.build_examples(raw_data)
        data = Dataset(examples)
        data_loader = data.create_batches(batch_size, shuffle, device)
        return data_loader





