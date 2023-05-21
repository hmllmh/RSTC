"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

from asyncore import write
import enum
from itertools import count
import os
from pickletools import optimize
from re import I
import time
import numpy as np
from sklearn import cluster
from sklearn.covariance import log_likelihood

from utils.logger import statistics_log
from utils.metric import Confusion
from dataloader.dataloader import unshuffle_loader

import torch
import torch.nn as nn
from torch.nn import functional as F
from learner.cluster_utils import target_distribution
from learner.contrastive_utils import PairConLoss
import ot
import sinkhornknopp as sk

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter


class SCCLvTrainer(nn.Module):
    def __init__(self, model, tokenizer, optimizer, train_loader, args, scheduler=None):
        super(SCCLvTrainer, self).__init__()
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.args = args
        self.eta = self.args.eta
        
        self.cluster_loss = nn.KLDivLoss(size_average=False)
        self.ce_loss = nn.CrossEntropyLoss()
        self.contrast_loss = PairConLoss(temperature=self.args.temperature, m=self.args.m)

        N = len(self.train_loader.dataset)
        self.a = torch.full((N, 1), 1/N).squeeze()

        self.b = torch.rand(self.args.classes, 1).squeeze()
        self.b = self.b / self.b.sum()
        
        self.u = None
        self.v = None
        self.h = torch.FloatTensor([1])
        self.allb = [[self.b[i].item()] for i in range(self.args.classes)]
        
        print(f"*****Intialize SCCLv, temp:{self.args.temperature}, eta:{self.args.eta}\n")

    def soft_ce_loss(self, pred, target, step):
        tmp = target ** 2 / torch.sum(target, dim=0)
        target = tmp / torch.sum(tmp, dim=1, keepdim=True)
        return torch.mean(-torch.sum(target * (F.log_softmax(pred, dim=1)), dim=1))

    def get_batch_token(self, text):
        token_feat = self.tokenizer.batch_encode_plus(
            text,
            max_length=self.args.max_length,
            return_tensors='pt', 
            padding='max_length', 
            truncation=True
        )
        return token_feat
        

    def prepare_transformer_input(self, batch):
        text1, text2, text3 = batch['text'], batch['augmentation_1'], batch['augmentation_2']
        feat1 = self.get_batch_token(text1)
        if self.args.augtype == 'explicit':
            feat2 = self.get_batch_token(text2)
            feat3 = self.get_batch_token(text3)
            input_ids = torch.cat([feat1['input_ids'].unsqueeze(1), feat2['input_ids'].unsqueeze(1), feat3['input_ids'].unsqueeze(1)], dim=1)
            attention_mask = torch.cat([feat1['attention_mask'].unsqueeze(1), feat2['attention_mask'].unsqueeze(1), feat3['attention_mask'].unsqueeze(1)], dim=1)
        else:
            input_ids = feat1['input_ids'] 
            attention_mask = feat1['attention_mask']
            
        return input_ids.cuda(), attention_mask.cuda()

    
    def loss_function(self, input_ids, attention_mask, selected, i):
        embd1, embd2, embd3 = self.model.get_embeddings(input_ids, attention_mask, task_type=self.args.augtype)

        # Instance-CL loss
        feat1, feat2 = self.model.contrast_logits(embd2, embd3)
        losses = self.contrast_loss(feat1, feat2)
        loss = self.eta * losses["loss"]
        losses['contrast'] = losses["loss"]
        self.args.tensorboard.add_scalar('loss/contrast_loss', losses['loss'].item(), global_step=i)

        # Clustering loss
        if i >= self.args.pre_step + 1:
            P1 = self.model(embd1)
            P2 = self.model(embd2)
            P3 = self.model(embd3)  # predicted labels before softmax
            target_label = None
            if len(self.L.shape) != 1:
                if self.args.soft == True:
                    target = self.L[selected].cuda()
                    cluster_loss = self.soft_ce_loss(P2, target, i) + self.soft_ce_loss(P3, target, i)
                else:
                    target_label = torch.argmax(self.L[selected], dim=1).cuda()
            else:
                target_label = self.L[selected].cuda()
            if target_label != None:
                cluster_loss = self.ce_loss(P2, target_label) + self.ce_loss(P3, target_label)
            loss += cluster_loss
            self.args.tensorboard.add_scalar('loss/cluster_loss', cluster_loss.item(), global_step=i)

            # 只训练cluster loss会帮助对比loss也变小，和有训练contrast loss时变化曲线相似
            # 只训练对比loss, cluster loss变小效果有限
            losses["cluster_loss"] = cluster_loss.item()
            
        losses['loss'] = loss
        self.args.tensorboard.add_scalar('loss/loss', loss, global_step=i)
        return loss, losses

    def train_step_explicit(self, input_ids, attention_mask, selected, i):
        if i >= self.optimize_times[-1]:
            _ = self.optimize_times.pop()
            if i >= self.args.pre_step + 40:
                self.get_labels(i)

        loss, losses = self.loss_function(input_ids, attention_mask, selected, i)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return losses

    def optimize_labels(self, step):
        # 1. aggregate P
        N = len(self.train_loader.dataset)
        PS = torch.zeros((N, self.args.classes))
        now = time.time()

        with torch.no_grad():
            for iter, (batch, selected) in enumerate(self.train_loader):
                input_ids, attention_mask = self.prepare_transformer_input(batch)
                emb1, _, _ = self.model.get_embeddings(input_ids, attention_mask, task_type=self.args.augtype)  # embedding
                p = F.softmax(self.model(emb1), dim=1)

                PS[selected] = p.detach().cpu()

        cost = -torch.log(PS)
        numItermax = 1000

        ###########
        if self.args.H == 'H2':
            #wang update b
            mu = 0.1
            z = torch.argmax(PS, dim=1)
            temp = list(range(self.args.num_classes))
            not_shown = list(set(temp).difference(set(z.numpy())))
            #print('not_shown ----:', len(not_shown))
            counts = Counter(z.numpy())
            for k in not_shown:
                counts[k] = 0
            self.b = mu * self.b + (1-mu) * torch.tensor(list(counts.values())) / sum(counts.values())
        #############
        T, log = sk.sinkhorn_knopp(self.a, self.b, cost, self.args.epsion, numItermax=numItermax, warn=False, log=True, u=self.u, v=self.v, h=self.h, reg2=self.args.reg2, log_alpha=self.args.logalpha, Hy=self.args.H)
        self.b = log['b']
        self.L = T
        print('Optimize Q takes {:.2f} min'.format((time.time() - now) / 60))
  
    def get_labels(self, step):
        # optimize labels
        print('[Step {}] Optimization starting'.format(step))
        # 更新self.L
        self.optimize_labels(step)
    def train(self):
        self.optimize_times = ((np.linspace(self.args.start, 1, self.args.M)**2)[::-1] * self.args.max_iter).tolist()
        # 训练前评估
        self.evaluate_embedding(-1)
        
        for i in np.arange(self.args.max_iter+1):
            self.model.train()
            try:
                batch, selected = next(train_loader_iter)
            except:
                train_loader_iter = iter(self.train_loader)
                batch, selected = next(train_loader_iter)

            input_ids, attention_mask = self.prepare_transformer_input(batch)
            losses = self.train_step_explicit(input_ids, attention_mask, selected, i) 

            if (self.args.print_freq>0) and ((i%self.args.print_freq==0) or (i==self.args.max_iter)):
                statistics_log(self.args.tensorboard, losses=losses, global_step=i)
                flag = self.evaluate_embedding(i)
                if flag == -1:
                    break
        return None

    def evaluate_embedding(self, step):
        dataloader = unshuffle_loader(self.args)
        print('---- {} evaluation batches ----'.format(len(dataloader)))
        
        self.model.eval()
        for i, batch in enumerate(dataloader):
            with torch.no_grad():
                text, label = batch['text'], batch['label'] 
                feat = self.get_batch_token(text)
                embeddings = self.model.get_embeddings(feat['input_ids'].cuda(), feat['attention_mask'].cuda(), task_type="evaluate")
                pred = torch.argmax(self.model(embeddings), dim=1)

                if i == 0:
                    all_labels = label
                    all_embeddings = embeddings.detach()
                    all_pred = pred.detach()
                else:
                    all_labels = torch.cat((all_labels, label), dim=0)
                    all_embeddings = torch.cat((all_embeddings, embeddings.detach()), dim=0)
                    all_pred = torch.cat((all_pred, pred.detach()), dim=0)

        # Initialize confusion matrices
        confusion = Confusion(max(self.args.num_classes, self.args.classes))
        rconfusion = Confusion(self.args.num_classes)
        embeddings = all_embeddings.cpu().numpy()
        pred_labels = all_pred.cpu()

        if step <= self.args.pre_step:
            kmeans = cluster.KMeans(n_clusters=self.args.classes, random_state=self.args.seed, max_iter=3000, tol=0.01)
            kmeans.fit(embeddings)
            kpred_labels = torch.tensor(kmeans.labels_.astype(np.int))
            self.L = kpred_labels
            pred_labels = kpred_labels
        # clustering accuracy
        clusters_num = len(set(pred_labels.numpy()))
        print('preded classes number:', clusters_num)
        self.args.tensorboard.add_scalar('Test/preded_clusters', clusters_num, step)
        confusion.add(pred_labels, all_labels)
        _, _ = confusion.optimal_assignment(self.args.num_classes)

        acc = confusion.acc()
        clusterscores = confusion.clusterscores(all_labels, pred_labels)

        ressave = {"acc":acc}
        ressave.update(clusterscores)
        for key, val in ressave.items():
            self.args.tensorboard.add_scalar('Test/{}'.format(key), val, step)

        print('[Step]', step)
        print('[Model] Clustering scores:', clusterscores) 
        print('[Model] ACC: {:.4f}'.format(acc))  # 使用kmeans聚类学到的representation的 acc

        # 停止标准：两次得到的label变化的占比少于tol
        y_pred = pred_labels.numpy()
        if step == -1:
            self.y_pred_last = np.copy(y_pred)
        else:
            change_rate = np.sum(y_pred != self.y_pred_last).astype(np.float32) / y_pred.shape[0]
            self.args.tensorboard.add_scalar('Test/change_rate', change_rate, step)
            self.y_pred_last = np.copy(y_pred)
            print('[Step] {} Label change rate: {:.3f} tol: {:.3f}'.format(step, change_rate, self.args.tol))
            if (step > self.args.pre_step and change_rate < self.args.tol) or step >= 3000:
                print('Reached tolerance threshold, stop training.')
                return -1
        return None