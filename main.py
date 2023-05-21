"""
Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved

Author: Dejiao Zhang (dejiaoz@amazon.com)
Date: 02/26/2021
"""

import os
from sched import scheduler
import sys
sys.path.append( './' )
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import argparse
from models.Transformers import SCCLBert
import dataloader.dataloader as dataloader
from training import SCCLvTrainer
from utils.kmeans import get_kmeans_centers
from utils.logger import setup_path, set_global_random_seed
from utils.optimizer import get_optimizer, get_bert
import numpy as np
import time


def run(args):
    args.resPath, args.tensorboard = setup_path(args)
    set_global_random_seed(args.seed)

    # dataset loader
    train_loader = dataloader.augmentation_loader(args)
    epochs = int(args.max_iter / (len(train_loader.dataset) / args.batch_size))
    if args.M == 0:
        args.M = epochs
        print(args.M)

    # model
    torch.cuda.set_device(args.gpuid[0])
    bert, tokenizer = get_bert(args)

    model = SCCLBert(bert, tokenizer, num_classes=args.num_classes, classes=args.classes, batch_size= args.batch_size)
    model = model.cuda()

    # optimizer 
    optimizer = get_optimizer(model, args)
    
    trainer = SCCLvTrainer(model, tokenizer, optimizer, train_loader, args, scheduler=None)
    trainer.train()
    
    return None

def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_instance', type=str, default='local') 
    parser.add_argument('--gpuid', nargs="+", type=int, default=[0], help="The list of gpuid, ex:--gpuid 3 1. Negative value means cpu-only")
    parser.add_argument('--seed', type=int, default=0, help="")
    parser.add_argument('--print_freq', type=float, default=50, help="")
    parser.add_argument('--resdir', type=str, default='./results/')
    parser.add_argument('--s3_resdir', type=str, default='./results')
    
    parser.add_argument('--bert', type=str, default='distilbert', help="")
    parser.add_argument('--use_pretrain', type=str, default='SBERT', choices=["BERT", "SBERT", "PAIRSUPCON"])
    
    # Dataset
    parser.add_argument('--datapath', type=str, default='./AugData/augmented-datasets')
    parser.add_argument('--dataname', type=str, default='tweet-original-order_trans_subst_10', help="")
    parser.add_argument('--num_classes', type=int, default=89, help="")
    parser.add_argument('--classes', type=int, default=89, help="")
    parser.add_argument('--max_length', type=int, default=32)
    parser.add_argument('--label', type=str, default='label')
    parser.add_argument('--text', type=str, default='text')
    parser.add_argument('--augmentation_1', type=str, default='text1')
    parser.add_argument('--augmentation_2', type=str, default='text2')
    # Learning parameters
    parser.add_argument('--lr', type=float, default=5e-6, help="")
    parser.add_argument('--lr_scale', type=int, default=100, help="")
    parser.add_argument('--max_iter', type=int, default=4000)
    parser.add_argument('--augtype', type=str, default='explicit', choices=['virtual', 'explicit'])
    parser.add_argument('--batch_size', type=int, default=200)
    # contrastive learning
    parser.add_argument('--temperature', type=float, default=1, help="temperature required by contrastive loss")
    parser.add_argument('--eta', type=float, default=10, help="")
    parser.add_argument('--m', type=float, default=0, help="additive angular margin penalty")

    # self labeling
    parser.add_argument('--soft', type=bool, default=False)
    parser.add_argument('--epsion', type=float, default=0.1, help="weight of enropy regularization")
    parser.add_argument('--reg2', type=float, default=0.1, help="weight of b contraint")
    parser.add_argument('--M', type=int, default=110)  # label assignment次数 与maxiter有关
    parser.add_argument('--tol', type=float, default=1e-2, help="change rate of label assignment between two evaluation")
    parser.add_argument('--start', type=float, default=0)
    parser.add_argument('--logalpha', type=int, default=100)

    # 对于不好的初始embedding，先仅用contrastive loss进行预训练一定的step
    parser.add_argument('--pre_step', type=int, default=600)
    parser.add_argument('--end', type=int, default=2000)
    # 不同的b的更新方式 H1 for ours; H2 for wang ; H3 for former
    parser.add_argument('--H', type=str, default='H1', choices=['H1', 'H2', 'H3'])
    
    args = parser.parse_args(argv)
    args.use_gpu = args.gpuid[0] >= 0
    args.resPath = None
    args.tensorboard = None

    return args



if __name__ == '__main__':
    
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))) 
    torch.cuda.empty_cache()
    import subprocess
       
    args = get_args(sys.argv[1:])

    if args.train_instance == "sagemaker":
        run(args)
        subprocess.run(["aws", "s3", "cp", "--recursive", args.resdir, args.s3_resdir])
    else:
        run(args)
    
    print(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))) 
            



    
