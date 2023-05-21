"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class SCCLBert(nn.Module):
    def __init__(self, bert_model, tokenizer, num_classes=8, classes=500, epsion=1, batch_size=200):
        super(SCCLBert, self).__init__()
        
        self.tokenizer = tokenizer
        self.bert = bert_model
        self.emb_size = self.bert.config.hidden_size
        self.epsion = epsion
        self.num_classes = num_classes
        self.classes = classes
        self.batch_size = batch_size

        # Instance-CL head
        self.contrast_head = nn.Sequential(  # for marginal cl  0.819
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.emb_size, 128)
            )
        self.adapt = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(self.emb_size, self.emb_size),
            nn.ReLU(inplace=True)
        )
        
        self.prob = nn.Linear(self.emb_size, self.classes)

    def init_weight(self):
        for y, m in enumerate(self.classifier):
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    
    def get_embeddings(self, input_ids, attention_mask, task_type="virtual"):
        if task_type == "evaluate":
            emb = self.get_mean_embeddings(input_ids, attention_mask)
            return emb
        
        elif task_type == "virtual":
            mean_output_1 = self.get_mean_embeddings(input_ids, attention_mask)
            mean_output_2 = self.get_mean_embeddings(input_ids, attention_mask)
            mean_output_3 = self.get_mean_embeddings(input_ids, attention_mask)
            return mean_output_1, mean_output_2, mean_output_3
        
        elif task_type == "explicit":
            input_ids_1, input_ids_2, input_ids_3 = torch.unbind(input_ids, dim=1)
            attention_mask_1, attention_mask_2, attention_mask_3 = torch.unbind(attention_mask, dim=1)

            mean_output_1 = self.get_mean_embeddings(input_ids_1, attention_mask_1)
            mean_output_2 = self.get_mean_embeddings(input_ids_2, attention_mask_2)
            mean_output_3 = self.get_mean_embeddings(input_ids_3, attention_mask_3)

            return mean_output_1, mean_output_2, mean_output_3
        
        else:
            raise Exception("TRANSFORMER ENCODING TYPE ERROR! OPTIONS: [EVALUATE, VIRTUAL, EXPLICIT]")
        
    def forward(self, embeddings):
        return self.prob(self.adapt(embeddings))
    
    def get_mean_embeddings(self, input_ids, attention_mask):
        bert_output = self.bert.forward(input_ids=input_ids, attention_mask=attention_mask)
        attention_mask = attention_mask.unsqueeze(-1)
        mean_output = torch.sum(bert_output[0]*attention_mask, dim=1) / torch.sum(attention_mask, dim=1)
        return mean_output

    def local_consistency(self, embd0, embd1, embd2, criterion):
        p0 = self.get_cluster_prob(embd0)
        p1 = self.get_cluster_prob(embd1)
        p2 = self.get_cluster_prob(embd2)
        
        lds1 = criterion(p1, p0)
        lds2 = criterion(p2, p0)
        return lds1+lds2

    def contrast_logits(self, embd1, embd2=None):
        c1 = self.contrast_head(embd1)
        feat1 = F.normalize(c1, dim=1)
        if embd2 != None:
            c2 = self.contrast_head(embd2)
            feat2 = F.normalize(c2, dim=1)
            return feat1, feat2
        else: 
            return feat1