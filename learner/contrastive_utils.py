"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 12/12/2021
"""


from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np


class PairConLoss(nn.Module):
    def __init__(self, temperature=0.05, m=0.4):
        super(PairConLoss, self).__init__()
        self.temperature = temperature
        self.eps = 1e-08
        self.m = m
        print(f"\n Initializing PairConLoss \n")

    def forward(self, features_1, features_2):
        loss = self.MCL(features_1, features_2, self.m, self.temperature)
        return {"loss":loss}

    def MCL(self, feature1, feature2, m, t):
        cos_theta12 = torch.matmul(feature1, feature2.T)  # N * N
        cos_theta_diag12 = torch.diag(cos_theta12)  # ii: N * 1
        theta12 = torch.acos(cos_theta_diag12)
        cos_theta_diag_m12 = torch.cos(theta12 + m)  # ii：positve pairs N * 1
        cos_theta11 = torch.matmul(feature1, feature1.T)  # 对角线元素都不用，其他的作为负样本
        cos_theta22 = torch.matmul(feature2, feature2.T)
        cos_theta21 = cos_theta12.T
        neg12 = torch.sum(torch.exp(cos_theta12 / t), dim=1) - torch.exp(cos_theta_diag12 / t)
        neg11 = torch.sum(torch.exp(cos_theta11 / t), dim=1) - torch.exp(torch.diag(cos_theta11) / t)
        neg22 = torch.sum(torch.exp(cos_theta22 / t), dim=1) - torch.exp(torch.diag(cos_theta22) / t)
        neg21 = torch.sum(torch.exp(cos_theta21 / t), dim=1) - torch.exp(torch.diag(cos_theta21) / t)
        cml_loss = -torch.mean(torch.log(torch.exp(cos_theta_diag_m12 / t) / (neg12 + neg11 + torch.exp(cos_theta_diag_m12 / t)))) - torch.mean(torch.log(torch.exp(cos_theta_diag_m12 / t) / (neg21 + neg22 + torch.exp(cos_theta_diag_m12 / t))))
        return cml_loss