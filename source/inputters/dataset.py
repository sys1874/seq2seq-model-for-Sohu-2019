#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/inputters/dataset.py
"""

import torch
from torch.utils.data import DataLoader

from source.utils.misc import Pack
from source.utils.misc import list2tensor
from source.utils.misc import map2tensor, align2tensor

class Dataset(torch.utils.data.Dataset):
    """
    Dataset
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def collate_fn(device=-1):
        """
        collate_fn
        """
        def collate(data_list):
            """
            collate
            """
            batch = Pack()
            for key in data_list[0].keys():
                batch[key] = list2tensor([x[key] for x in data_list])
            if device >= 0:
                batch = batch.cuda(device=device)
            return batch
        return collate

    def create_batches(self, batch_size=1, shuffle=False, device=-1):
        """
        create_batches
        """
        loader = DataLoader(dataset=self,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            collate_fn=self.collate_fn(device),
                            pin_memory=False)
        return loader

#!/usr/bin/env python
# -*- coding: UTF-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/inputters/dataset.py
"""



class Entity_Dataset(torch.utils.data.Dataset):
    """
    Dataset
    """
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def collate_fn(device=-1):
        """
        collate_fn
        """
        def collate(data_list):
            """
            collate
            """
            batch = Pack()
            # 手写各个结构
            # num_src
            batch['num_src']=list2tensor([x['num_src'] for x in data_list])
            # num_tgt_input
            batch['num_tgt_input'] = list2tensor([x['num_tgt_input'] for x in data_list])
            #  tgt_output
            batch['tgt_output'] = list2tensor([x['tgt_output'] for x in data_list])
            batch['tgt_emo'] =list2tensor([x['tgt_emo'] for x in data_list])
            # mask
            batch['mask'] = list2tensor([x['mask'] for x in data_list])
            batch['raw_src']=[x['raw_src'] for x in data_list]
            batch['raw_tgt'] = [x['raw_tgt'] for x in data_list]

            if 'id' in data_list[0].keys():
                batch['id']=[x['id'] for x in data_list]

            if device >= 0:
                batch = batch.cuda(device=device)
            return batch
        return collate

    def create_batches(self, batch_size=1, shuffle=False, device=-1):
        """
        create_batches
        """
        loader = DataLoader(dataset=self,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            collate_fn=self.collate_fn(device),
                            pin_memory=False)
        return loader


class Entity_Dataset_pos(torch.utils.data.Dataset):
    """
    Dataset
    """

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def collate_fn(device=-1):
        """
        collate_fn
        """

        def collate(data_list):
            """
            collate
            """
            batch = Pack()
            # 手写各个结构
            # num_src
            batch['num_src'] = list2tensor([x['num_src'] for x in data_list])
            batch['num_pos'] = list2tensor([x['num_pos'] for x in data_list])
            # num_tgt_input
            batch['num_tgt_input'] = list2tensor([x['num_tgt_input'] for x in data_list])
            #  tgt_output
            batch['tgt_output'] = list2tensor([x['tgt_output'] for x in data_list])
            batch['tgt_emo'] = list2tensor([x['tgt_emo'] for x in data_list])
            # mask
            batch['mask'] = list2tensor([x['mask'] for x in data_list])
            batch['raw_src'] = [x['raw_src'] for x in data_list]
            batch['raw_tgt'] = [x['raw_tgt'] for x in data_list]

            if 'id' in data_list[0].keys():
                batch['id'] = [x['id'] for x in data_list]

            if device >= 0:
                batch = batch.cuda(device=device)
            return batch

        return collate

    def create_batches(self, batch_size=1, shuffle=False, device=-1):
        """
        create_batches
        """
        loader = DataLoader(dataset=self,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            collate_fn=self.collate_fn(device),
                            pin_memory=False)
        return loader
