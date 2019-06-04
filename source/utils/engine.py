#!/usr/bin/env python
# -*- coding: utf-8 -*-
################################################################################
#
# Copyright (c) 2019 Baidu.com, Inc. All Rights Reserved
#
################################################################################
"""
File: source/utils/engine.py
"""

import os
import time
import shutil
import numpy as np
import torch
from collections import  Counter

from collections import defaultdict
# from tensorboardX import SummaryWriter

from source.utils.metrics import bleu, distinct
from source.utils.metrics import EmbeddingMetrics


class MetricsManager(object):
    """
    MetricsManager
    """
    def __init__(self):
        self.metrics_val = defaultdict(float)
        self.metrics_cum = defaultdict(float)
        self.num_samples = 0

    def update(self, metrics):
        """
        update
        """
        num_samples = metrics.pop("num_samples", 1)
        self.num_samples += num_samples

        for key, val in metrics.items():
            if val is not None:
                if isinstance(val, torch.Tensor):
                    val = val.item()
                    self.metrics_cum[key] += val * num_samples
                else:
                    assert len(val) == 2
                    val, num_words = val[0].item(), val[1]
                    self.metrics_cum[key] += np.array(
                        [val * num_samples, num_words])
                self.metrics_val[key] = val

    def clear(self):
        """
        clear
        """
        self.metrics_val = defaultdict(float)
        self.metrics_cum = defaultdict(float)
        self.num_samples = 0

    def get(self, name):
        """
        get
        """
        val = self.metrics_cum.get(name)
        if not isinstance(val, float):
            val = val[0]
        return val / self.num_samples

    def report_val(self):
        """
        report_val
        """
        metric_strs = []
        for key, val in self.metrics_val.items():
            metric_str = "{}-{:.3f}".format(key.upper(), val)
            metric_strs.append(metric_str)
        metric_strs = "   ".join(metric_strs)
        return metric_strs

    def report_cum(self):
        """
        report_cum
        """
        metric_strs = []
        for key, val in self.metrics_cum.items():
            if isinstance(val, float):
                val, num_words = val, None
            else:
                val, num_words = val

            metric_str = "{}-{:.3f}".format(key.upper(), val / self.num_samples)
            metric_strs.append(metric_str)

            if num_words is not None:
                ppl = np.exp(min(val / num_words, 100))
                metric_str = "{}_PPL-{:.3f}".format(key.upper(), ppl)
                metric_strs.append(metric_str)

        metric_strs = "   ".join(metric_strs)
        return metric_strs


def evaluate(model, data_iter, verbose=False):
    """
    evaluate
    """
    model.eval()
    mm = MetricsManager()
    with torch.no_grad():
        for inputs in data_iter:
            metrics= model.iterate(inputs=inputs, is_training=False)
            mm.update(metrics)
    return mm


class Trainer(object):
    """
    Trainer
    """
    def __init__(self,
                 model,
                 optimizer,
                 train_iter,
                 valid_iter,
                 logger,
                 generator=None,
                 valid_metric_name="acc",
                 num_epochs=1,
                 save_dir=None,
                 log_steps=None,
                 valid_steps=None,
                 grad_clip=None,
                 lr_scheduler=None,
                 save_summary=False,
                 ):
        self.model = model
        self.optimizer = optimizer
        self.train_iter = train_iter
        self.valid_iter = valid_iter
        self.logger = logger

        self.generator = generator
        self.is_decreased_valid_metric = False
        self.valid_metric_name = valid_metric_name
        self.num_epochs = num_epochs
        self.save_dir = save_dir
        self.log_steps = log_steps
        self.valid_steps = valid_steps
        self.grad_clip = grad_clip
        self.lr_scheduler = lr_scheduler
        self.save_summary = save_summary


        # if self.save_summary:
        #     self.train_writer = SummaryWriter(
        #         os.path.join(self.save_dir, "logs", "train"))
        #     self.valid_writer = SummaryWriter(
        #         os.path.join(self.save_dir, "logs", "valid"))

        self.best_valid_metric = float(
            "inf") if self.is_decreased_valid_metric else -float("inf")
        self.epoch = 0
        self.batch_num = 0

        self.train_start_message = "\n".join(["",
                                              "=" * 85,
                                              "=" * 34 + " Model Training " + "=" * 35,
                                              "=" * 85,
                                              ""])
        self.valid_start_message = "\n" + "-" * 33 + " Model Evaulation " + "-" * 33

    def summarize_train_metrics(self, metrics, global_step):
        """
        summarize_train_metrics
        """
        for key, val in metrics.items():
            if isinstance(val, (list, tuple)):
                val = val[0]
            if isinstance(val, torch.Tensor):
                self.train_writer.add_scalar(key, val, global_step)

    def summarize_valid_metrics(self, metrics_mm, global_step):
        """
        summarize_valid_metrics
        """
        for key in metrics_mm.metrics_cum.keys():
            val = metrics_mm.get(key)
            self.valid_writer.add_scalar(key, val, global_step)

    def train_epoch(self):
        """
        train_epoch
        """
        self.epoch += 1

        # if self.epoch>=2:
        #     valid_steps=510
        #     if self.epoch>2:
        #         for group in self.optimizer.param_groups:
        #             group['lr']=group['lr']*0.5
        # if self.epoch>=4:
        #     valid_steps=210
        #     if self.epoch>4:
        #         for group in self.optimizer.param_groups:
        #             group['lr']=group['lr']*0.5
        # else:
        #     valid_steps=self.valid_steps
        valid_steps = self.valid_steps
        train_mm = MetricsManager()
        num_batches = len(self.train_iter)
        self.logger.info(self.train_start_message)

        for batch_id, inputs in enumerate(self.train_iter, 1):
            self.model.train()
            # if self.model.encoder.elmo_embedder is not None:
            #     self.model.encoder.elmo_embedder.model.eval()

            start_time = time.time()
            # Do a training iteration
            metrics= self.model.iterate(inputs,
                                         optimizer=self.optimizer,
                                         grad_clip=self.grad_clip,
                                         is_training=True,
                                         epoch=self.epoch)
            elapsed = time.time() - start_time

            train_mm.update(metrics)
            self.batch_num += 1

            if batch_id % self.log_steps == 0:
                message_prefix = "[Train][{:2d}][{}/{}]".format(self.epoch, batch_id, num_batches)
                metrics_message = train_mm.report_val()
                message_posfix = "TIME-{:.2f}".format(elapsed)
                self.logger.info("   ".join(
                    [message_prefix, metrics_message, message_posfix]))
                if self.save_summary:
                    self.summarize_train_metrics(metrics, self.batch_num)

            if batch_id % valid_steps == 0:
                self.logger.info(self.valid_start_message)
                valid_mm = evaluate(self.model, self.valid_iter)

                message_prefix = "[Valid][{:2d}][{}/{}]".format(self.epoch, batch_id, num_batches)
                metrics_message = valid_mm.report_cum()
                self.logger.info("   ".join([message_prefix, metrics_message]))

                gen_eval_metrics, F1 = evaluate_generation(generator=self.generator,
                                                          data_iter=self.valid_iter,
                                                          save_file=None)
                self.logger.info(gen_eval_metrics)

                if self.save_summary:
                    self.summarize_valid_metrics(valid_mm, self.batch_num)
                cur_valid_metric=F1
                # cur_valid_metric = valid_mm.get(self.valid_metric_name)
                if self.is_decreased_valid_metric:
                    is_best = cur_valid_metric < self.best_valid_metric
                else:
                    is_best = cur_valid_metric > self.best_valid_metric
                if is_best:
                    self.best_valid_metric = cur_valid_metric
                self.save(is_best)
                if self.lr_scheduler is not None and (batch_id%self.valid_steps)==0:
                    self.lr_scheduler.step(valid_mm.get('loss'))
                self.logger.info("-" * 85 + "\n")

        if self.generator is not None:
            self.logger.info("Generation starts ...")
            gen_save_file = os.path.join(
                self.save_dir, "valid_{}.result").format(self.epoch)
            gen_eval_metrics, F1 = evaluate_generation(generator=self.generator,
                                                   data_iter=self.valid_iter,
                                                   save_file=gen_save_file)
            self.logger.info(gen_eval_metrics)
            is_best = F1 > self.best_valid_metric
            if is_best:
                self.best_valid_metric=F1
            self.save(is_best)
        self.logger.info('')



    def train(self):
        """
        train
        """
        # valid_mm = evaluate(self.model, self.valid_iter)
        # self.logger.info(valid_mm.report_cum())
        for _ in range(self.epoch, self.num_epochs):
            self.train_epoch()

    def save(self, is_best=False):
        """
        save
        """
        model_file = os.path.join(
            self.save_dir, "state_epoch_{}.model".format(self.epoch))
        torch.save(self.model.state_dict(), model_file)
        self.logger.info("Saved model state to '{}'".format(model_file))

        train_file = os.path.join(
            self.save_dir, "state_epoch_{}.train".format(self.epoch))
        train_state = {"epoch": self.epoch,
                       "batch_num": self.batch_num,
                       "best_valid_metric": self.best_valid_metric,
                       "optimizer": self.optimizer.state_dict()}
        if self.lr_scheduler is not None:
            train_state["lr_scheduler"] = self.lr_scheduler.state_dict()
        torch.save(train_state, train_file)
        self.logger.info("Saved train state to '{}'".format(train_file))

        if is_best:
            best_model_file = os.path.join(self.save_dir, "best.model")
            best_train_file = os.path.join(self.save_dir, "best.train")
            shutil.copy(model_file, best_model_file)
            shutil.copy(train_file, best_train_file)
            self.logger.info(
                "Saved best model state to '{}' with new best valid metric {}-{:.3f}".format(
                    best_model_file, self.valid_metric_name.upper(), self.best_valid_metric))

    def load(self, file_prefix):
        """
        load
        """
        model_file = "{}.model".format(file_prefix)
        train_file = "{}.train".format(file_prefix)

        model_state_dict = torch.load(
            model_file, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(model_state_dict)
        self.logger.info("Loaded model state from '{}'".format(model_file))

        train_state_dict = torch.load(
            train_file, map_location=lambda storage, loc: storage)
        self.epoch = train_state_dict["epoch"]
        self.best_valid_metric = train_state_dict["best_valid_metric"]
        self.batch_num = train_state_dict["batch_num"]
        self.optimizer.load_state_dict(train_state_dict["optimizer"])
        if self.lr_scheduler is not None and "lr_scheduler" in train_state_dict:
            self.lr_scheduler.load_state_dict(train_state_dict["lr_scheduler"])
        self.logger.info(
            "Loaded train state from '{}' with (epoch-{} best_valid_metric-{:.3f})".format(
                train_file, self.epoch, self.best_valid_metric))


def evaluate_generation(generator,
                        data_iter,
                        save_file=None,
                        num_batches=None,
                        verbos=False,
                        for_test=False):
    """
    evaluate_generation
    """
    results = generator.generate(batch_iter=data_iter,
                                 num_batches=num_batches)

    # refs = [result.tgt.split(" ") for result in results]
    # hyps = [result.preds[0].split(" ") for result in results]
    refs = [result.tgt[1:] for result in results]
    hyps = [result.preds[0] for result in results]

    emo_refs=[result.target_emos[:-1] for result in results]
    emo_hyps=[result.emos[0] for result in results]

    temp_ref=[]
    for a,b in zip(refs,emo_refs):
        temp_ref.append(dict(zip(a,b)))

    temp_hyp = []
    for a, b in zip(hyps, emo_hyps):
        temp_hyp.append(dict(zip(a, b)))

    p=0
    t=0
    for d1, d2 in zip(temp_hyp, temp_ref):
        t+=len(d2)
        for key in d1:
            if key in d2:
                if d1[key]==d1[key]:
                    p+=1
    emo_p=p/(t+1e-15)






    F1=0
    precision=0
    recal=0
    for ref, hyp in zip(refs,hyps):
        r_c=Counter(ref)
        h_c=Counter(hyp)

        common =r_c & h_c
        total = sum(common.values())
        r_l = sum(r_c.values())
        h_l = sum(h_c.values())
        p = total / (h_l+1e-15)
        r = total / (r_l+1e-15)
        f1 = 2 * p * r / (p + r+1e-15)
        precision+=p
        recal+=r
        F1+=f1
    F1=F1/len(hyps)
    precision = precision / len(hyps)
    recal = recal / len(hyps)




    report_message = []

    avg_len = np.average([len(s) for s in hyps])
    report_message.append("Avg_Len-{:.3f}".format(avg_len))
    report_message.append("F1-{:.3f}".format(F1))
    report_message.append("precision-{:.3f}".format(precision))
    report_message.append("recal-{:.3f}".format(recal))
    report_message.append("emo_recall-{:.3f}".format(emo_p))
    report_message = "   ".join(report_message)

    # intra_dist1, intra_dist2, inter_dist1, inter_dist2 = distinct(refs)
    avg_len = np.average([len(s) for s in refs])
    target_message = "Target:   AVG_LEN-{:.3f}   ".format(avg_len)
    # target_message = "Target:   AVG_LEN-{:.3f}   ".format(avg_len) + \
    #     "Inter_Dist-{:.4f}/{:.4f}".format(inter_dist1, inter_dist2)

    # message = report_message + "\n" + target_message
    message = report_message

    if save_file is not None:
        if not for_test:
            write_results(temp_ref, temp_hyp, save_file)
            print("Saved generation results to '{}'".format(save_file))
        else:
            write_results_pre(hyps, emo_hyps, results,  save_file)
    if verbos:
        print(message+'\n'+target_message)
    else:
        return message, F1


def write_results_pre(hyps, emo_hyps, results,  save_file):
    new_id=[x['id'] for x in results]
    with open(save_file, 'w', encoding='utf-8') as wf:
        for id, entity, emotion in zip(new_id, hyps, emo_hyps):
            entity_temp=[]
            emotion_temp=[]
            e_s=set()
            for i , word in enumerate(entity):
                if word not in e_s:
                    e_s.add(word)
                    entity_temp.append(word)
                    emotion_temp.append(emotion[i])
            entity=entity_temp
            emotion=emotion_temp
            wf.write( id + '\t' + ','.join(entity) + '\t' + ','.join(emotion) + '\n')
        wf.close()

def write_results(temp_ref, temp_hyp, save_file):
    with open(save_file, 'w', encoding='utf-8') as wf:
        for d1, d2 in zip(temp_hyp, temp_ref):
            wf.write('pre:'+str(d1)+'\n')
            wf.write('tgt:'+str(d2) + '\n')
        wf.close()

