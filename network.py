#!/usr/bin/env python
# -*- coding: utf-8 -*-
######################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
# @file network.py
#
######################################################################
"""
File: network.py
"""

import os
import sys
import json
import shutil
import logging
import argparse
import torch
from datetime import datetime

from source.inputters.corpus import Entity_Corpus, Entity_Corpus_pos
from source.models.Entity_seq2seq import Entity_Seq2Seq
from source.models.Entity_seq2seq_pos import Entity_Seq2Seq_pos
from source.models.Entity_seq2seq_pos_gru import Entity_Seq2Seq_pos_gru
from source.models.Entity_seq2seq_elmo import Entity_Seq2Seq_elmo
from source.models.Entity_seq2seq_elmo_gru import Entity_Seq2Seq_elmo_gru


from source.utils.engine import Trainer
from source.utils.generator import TopKGenerator
from source.utils.engine import evaluate, evaluate_generation
from source.utils.misc import str2bool

def model_config():
    """
    model_config
    """
    parser = argparse.ArgumentParser()

    # Data
    data_arg = parser.add_argument_group("Data")
    data_arg.add_argument("--data_dir", type=str, default="./data/")
    data_arg.add_argument("--data_prefix", type=str, default="demo")
    data_arg.add_argument("--save_dir", type=str, default="./models/")
    data_arg.add_argument("--with_label", type=str2bool, default=False)
    data_arg.add_argument("--embed_file", type=str, default=None)

    # Network
    net_arg = parser.add_argument_group("Network")
    net_arg.add_argument("--embed_size", type=int, default=300)
    net_arg.add_argument("--hidden_size", type=int, default=800)
    net_arg.add_argument("--bidirectional", type=str2bool, default=True)
    net_arg.add_argument("--max_vocab_size", type=int, default=30000)
    net_arg.add_argument("--min_len", type=int, default=1)
    net_arg.add_argument("--max_len", type=int, default=500)
    net_arg.add_argument("--num_layers", type=int, default=1)
    net_arg.add_argument("--attn", type=str, default='dot',
                         choices=['none', 'mlp', 'dot', 'general'])
    net_arg.add_argument("--share_vocab", type=str2bool, default=True)
    net_arg.add_argument("--with_bridge", type=str2bool, default=True)
    net_arg.add_argument("--tie_embedding", type=str2bool, default=True)

    # Training / Testing
    train_arg = parser.add_argument_group("Training")
    train_arg.add_argument("--optimizer", type=str, default="Adam")
    train_arg.add_argument("--lr", type=float, default=0.00005)
    train_arg.add_argument("--grad_clip", type=float, default=5.0)
    train_arg.add_argument("--dropout", type=float, default=0.3)
    train_arg.add_argument("--num_epochs", type=int, default=20)
    train_arg.add_argument("--pretrain_epoch", type=int, default=5)
    train_arg.add_argument("--lr_decay", type=float, default=None)
    train_arg.add_argument("--use_embed", type=str2bool, default=True)
    train_arg.add_argument("--use_bow", type=str2bool, default=True)
    train_arg.add_argument("--use_dssm", type=str2bool, default=False)
    train_arg.add_argument("--use_pg", type=str2bool, default=False)
    train_arg.add_argument("--use_gs", type=str2bool, default=False)
    train_arg.add_argument("--use_kd", type=str2bool, default=False)
    train_arg.add_argument("--weight_control", type=str2bool, default=False)
    train_arg.add_argument("--decode_concat", type=str2bool, default=False)
    train_arg.add_argument("--use_posterior", type=str2bool, default=True)

    # Geneation
    gen_arg = parser.add_argument_group("Generation")
    gen_arg.add_argument("--max_dec_len", type=int, default=5)
    gen_arg.add_argument("--ignore_unk", type=str2bool, default=True)
    gen_arg.add_argument("--length_average", type=str2bool, default=True)
    gen_arg.add_argument("--gen_file", type=str, default="./test.result")
    gen_arg.add_argument("--gold_score_file", type=str, default="./gold.scores")

    # MISC
    misc_arg = parser.add_argument_group("Misc")
    misc_arg.add_argument("--gpu", type=int, default=-1)
    misc_arg.add_argument("--log_steps", type=int, default=20)
    misc_arg.add_argument("--valid_steps", type=int, default=20)
    misc_arg.add_argument("--batch_size", type=int, default=32)
    misc_arg.add_argument("--ckpt", type=str)
    #misc_arg.add_argument("--ckpt", type=str, default="models/best.model")
    misc_arg.add_argument("--check", action="store_true")
    misc_arg.add_argument("--test", action="store_true")
    misc_arg.add_argument("--interact", action="store_true")
    #misc_arg.add_argument("--interact", type=str2bool, default=True)
    misc_arg.add_argument("--preprocess", action='store_true', help="仅执行preprocess")
    misc_arg.add_argument("--entity_file", type=str, default=None, help='实体文件')
    misc_arg.add_argument('--min_freq' , type=int, default=0)
    misc_arg.add_argument('--for_test', action='store_true', help='修改生成的结果方式')
    misc_arg.add_argument('--beam_size', type=int, default=1)
    misc_arg.add_argument('--saved_embed', type=str, default=None, help='处理好的词向量')


    misc_arg.add_argument('--pos', action='store_true', help='使用pos')

    misc_arg.add_argument('--weight_decay', type=float ,default=0, help='增加 weight decay')

    misc_arg.add_argument('--rnn_type' ,type=str, default='lstm', help='编码类型')

    misc_arg.add_argument('--elmo' , action='store_true', help='使用elmo')




    config = parser.parse_args()

    return config


def main():
    """
    main
    """
    config = model_config()
    if config.check:
        config.save_dir = "./tmp/"
    config.use_gpu = torch.cuda.is_available() and config.gpu >= 0
    device = config.gpu
    torch.cuda.set_device(device)
    # Data definition
    if config.pos:
        corpus = Entity_Corpus_pos(data_dir=config.data_dir, data_prefix=config.data_prefix, entity_file=config.entity_file,
                                     min_freq=config.min_freq, max_vocab_size=config.max_vocab_size)

    else:
        corpus = Entity_Corpus(data_dir=config.data_dir, data_prefix=config.data_prefix,
                                   entity_file=config.entity_file,
                                   min_freq=config.min_freq, max_vocab_size=config.max_vocab_size)

    corpus.load()
    if config.test and config.ckpt:
        corpus.reload(data_type='test')
    train_iter = corpus.create_batches(
        config.batch_size, "train", shuffle=True, device=device)
    valid_iter = corpus.create_batches(
        config.batch_size, "valid", shuffle=False, device=device)
    if config.for_test:
        test_iter = corpus.create_batches(
            config.batch_size, "test", shuffle=False, device=device)
    else:
        test_iter = corpus.create_batches(
            config.batch_size, "valid", shuffle=False, device=device)
    if config.preprocess:
        print('预处理完毕')
        return

    if config.pos:
        if config.rnn_type == 'lstm':
            model = Entity_Seq2Seq_pos(src_vocab_size=corpus.SRC.vocab_size,
                                       pos_vocab_size=corpus.POS.vocab_size,
                                       embed_size=config.embed_size, hidden_size=config.hidden_size,
                                       padding_idx=corpus.padding_idx,
                                       num_layers=config.num_layers, bidirectional=config.bidirectional,
                                       attn_mode=config.attn, with_bridge=config.with_bridge,
                                       dropout=config.dropout,
                                       use_gpu=config.use_gpu,
                                       pretrain_epoch=config.pretrain_epoch)
        else:
            model = Entity_Seq2Seq_pos_gru(src_vocab_size=corpus.SRC.vocab_size,
                                           pos_vocab_size=corpus.POS.vocab_size,
                                           embed_size=config.embed_size, hidden_size=config.hidden_size,
                                           padding_idx=corpus.padding_idx,
                                           num_layers=config.num_layers, bidirectional=config.bidirectional,
                                           attn_mode=config.attn, with_bridge=config.with_bridge,
                                           dropout=config.dropout,
                                           use_gpu=config.use_gpu,
                                           pretrain_epoch=config.pretrain_epoch)
    else:
        if config.rnn_type == 'lstm':
            if config.elmo:
                model = Entity_Seq2Seq_elmo(src_vocab_size=corpus.SRC.vocab_size,
                                            embed_size=config.embed_size, hidden_size=config.hidden_size,
                                            padding_idx=corpus.padding_idx,
                                            num_layers=config.num_layers, bidirectional=config.bidirectional,
                                            attn_mode=config.attn, with_bridge=config.with_bridge,
                                            dropout=config.dropout,
                                            use_gpu=config.use_gpu,
                                            pretrain_epoch=config.pretrain_epoch,
                                            batch_size=config.batch_size)
            else:
                model = Entity_Seq2Seq(src_vocab_size=corpus.SRC.vocab_size,
                                       embed_size=config.embed_size, hidden_size=config.hidden_size,
                                       padding_idx=corpus.padding_idx,
                                       num_layers=config.num_layers, bidirectional=config.bidirectional,
                                       attn_mode=config.attn, with_bridge=config.with_bridge,
                                       dropout=config.dropout,
                                       use_gpu=config.use_gpu,
                                       pretrain_epoch=config.pretrain_epoch)
        else:  # GRU
            if config.elmo:
                model = Entity_Seq2Seq_elmo_gru(src_vocab_size=corpus.SRC.vocab_size,
                                                embed_size=config.embed_size, hidden_size=config.hidden_size,
                                                padding_idx=corpus.padding_idx,
                                                num_layers=config.num_layers, bidirectional=config.bidirectional,
                                                attn_mode=config.attn, with_bridge=config.with_bridge,
                                                dropout=config.dropout,
                                                use_gpu=config.use_gpu,
                                                pretrain_epoch=config.pretrain_epoch,
                                                batch_size=config.batch_size)

    # if config.pos:
    #     if config.rnn_type=='lstm':
    #         if config.elmo:
    #             model = Entity_Seq2Seq_elmo(src_vocab_size=corpus.SRC.vocab_size,
    #                                    embed_size=config.embed_size, hidden_size=config.hidden_size,
    #                                    padding_idx=corpus.padding_idx,
    #                                    num_layers=config.num_layers, bidirectional=config.bidirectional,
    #                                    attn_mode=config.attn, with_bridge=config.with_bridge,
    #                                    dropout=config.dropout,
    #                                    use_gpu=config.use_gpu,
    #                                    pretrain_epoch=config.pretrain_epoch,
    #                                    batch_size=config.batch_size)
    #         else:
    #             model = Entity_Seq2Seq_pos(src_vocab_size=corpus.SRC.vocab_size,
    #                                    pos_vocab_size=corpus.POS.vocab_size,
    #                                    embed_size=config.embed_size, hidden_size=config.hidden_size,
    #                                    padding_idx=corpus.padding_idx,
    #                                    num_layers=config.num_layers, bidirectional=config.bidirectional,
    #                                    attn_mode=config.attn, with_bridge=config.with_bridge,
    #                                    dropout=config.dropout,
    #                                    use_gpu=config.use_gpu,
    #                                    pretrain_epoch=config.pretrain_epoch)
    #     else:
    #         if config.elmo:
    #             model = Entity_Seq2Seq_elmo_gru(src_vocab_size=corpus.SRC.vocab_size,
    #                                    embed_size=config.embed_size, hidden_size=config.hidden_size,
    #                                    padding_idx=corpus.padding_idx,
    #                                    num_layers=config.num_layers, bidirectional=config.bidirectional,
    #                                    attn_mode=config.attn, with_bridge=config.with_bridge,
    #                                    dropout=config.dropout,
    #                                    use_gpu=config.use_gpu,
    #                                    pretrain_epoch=config.pretrain_epoch,
    #                                    batch_size=config.batch_size)
    #         else:
    #             model =Entity_Seq2Seq_pos_gru(src_vocab_size=corpus.SRC.vocab_size,
    #                                pos_vocab_size=corpus.POS.vocab_size,
    #                                embed_size=config.embed_size, hidden_size=config.hidden_size,
    #                                padding_idx=corpus.padding_idx,
    #                                num_layers=config.num_layers, bidirectional=config.bidirectional,
    #                                attn_mode=config.attn, with_bridge=config.with_bridge,
    #                                dropout=config.dropout,
    #                                use_gpu=config.use_gpu,
    #                                pretrain_epoch=config.pretrain_epoch)
    # else:
    #     model = Entity_Seq2Seq(src_vocab_size=corpus.SRC.vocab_size,
    #                              embed_size=config.embed_size, hidden_size=config.hidden_size,
    #                              padding_idx=corpus.padding_idx,
    #                              num_layers=config.num_layers, bidirectional=config.bidirectional,
    #                              attn_mode=config.attn, with_bridge=config.with_bridge,
    #                              dropout=config.dropout,
    #                              use_gpu=config.use_gpu,
    #                              pretrain_epoch=config.pretrain_epoch)



    model_name = model.__class__.__name__
    # Generator definition

    generator = TopKGenerator(model=model,
                                      src_field=corpus.SRC,
                                      max_length=config.max_dec_len, ignore_unk=config.ignore_unk,
                          length_average=config.length_average, use_gpu=config.use_gpu, beam_size=config.beam_size)
    # generator=None
    # Interactive generation testing
    if config.interact and config.ckpt:
        model.load(config.ckpt)
        return generator
    # Testing
    elif config.test and config.ckpt:
        print(model)
        model.load(config.ckpt)
        print("Testing ...")
        metrics = evaluate(model, valid_iter)
        print(metrics.report_cum())
        print("Generating ...")
        if config.for_test:
            evaluate_generation(generator, test_iter, save_file=config.gen_file, verbos=True, for_test=True)
        else:
            evaluate_generation(generator, test_iter, save_file=config.gen_file, verbos=True)
    else:
        # Load word embeddings
        if config.saved_embed is not None:
            model.encoder.embedder.load_embeddings(
                config.saved_embed, scale=0.03)
        # Optimizer definition
        # if config.saved_embed:
        #     embed=[]
        #     other=[]
        #     for name, v in model.named_parameters():
        #         if '.embedder' in name:
        #             print(name)
        #             embed.append(v)
        #         else:
        #             other.append(v)
        #     optimizer = getattr(torch.optim, config.optimizer)([{'params': other,
        #        'lr': config.lr,  'eps': 1e-8},
        #       {'params': embed,  'lr': config.lr/2, 'eps': 1e-8}])
        p=model.parameters()
        p=[x for x in p if x.requires_grad]
        optimizer = getattr(torch.optim, config.optimizer)(
            p, lr=config.lr, weight_decay=config.weight_decay)
        # Learning rate scheduler
        if config.lr_decay is not None and 0 < config.lr_decay < 1.0:
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, 
                            factor=config.lr_decay, patience=1, verbose=True, min_lr=1e-5)
        else:
            lr_scheduler = None
        # Save directory
        date_str, time_str = datetime.now().strftime("%Y%m%d-%H%M%S").split("-")
        result_str = "{}-{}".format(model_name, time_str)
        if not os.path.exists(config.save_dir):
            os.makedirs(config.save_dir)
        # Logger definition
        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.DEBUG, format="%(message)s")
        fh = logging.FileHandler(os.path.join(config.save_dir, "train.log"))
        logger.addHandler(fh)
        # Save config
        params_file = os.path.join(config.save_dir, "params.json")
        with open(params_file, 'w') as fp:
            json.dump(config.__dict__, fp, indent=4, sort_keys=True)
        print("Saved params to '{}'".format(params_file))
        logger.info(model)
        # Train
        logger.info("Training starts ...")
        trainer = Trainer(model=model, optimizer=optimizer, train_iter=train_iter,
                          valid_iter=valid_iter, logger=logger, generator=generator,
                          valid_metric_name="acc", num_epochs=config.num_epochs,
                          save_dir=config.save_dir, log_steps=config.log_steps,
                          valid_steps=config.valid_steps, grad_clip=config.grad_clip,
                          lr_scheduler=lr_scheduler, save_summary=False)
        if config.ckpt is not None:
            trainer.load(file_prefix=config.ckpt)
        trainer.train()
        logger.info("Training done!")
        # Test
        logger.info("")
        trainer.load(os.path.join(config.save_dir, "best"))
        logger.info("Testing starts ...")
        metrics, scores = evaluate(model, test_iter)
        logger.info(metrics.report_cum())
        logger.info("Generation starts ...")
        test_gen_file = os.path.join(config.save_dir, "test.result")
        evaluate_generation(generator, test_iter, save_file=test_gen_file, verbos=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\nExited from the program ealier!")
