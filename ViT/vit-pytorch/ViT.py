import torch
import numpy as np
from torch import nn
import torch.autograd.profiler as profiler
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import BertModel, BertConfig

from transformers import BertTokenizer, BertModel
import torch
import torch
import vit_pytorch
from vit_pytorch import ViT

"""***************** configuration here *******************"""

flag = 0 # 0: MLP profiling, 1: MLP end-to-end run time 
iter1 = 30
iter2 = 300
batch_size = 48
channel = 3
image_size = 252
dim = 1024
depth = 24
heads = 16
mlp_dim = 4096
dropout = 0.1
emb_dropout = 0.1
num_classes = 1000
patch_size = (36, 28)

"""***************** configuration end *******************"""

v = ViT(image_size = image_size,
        patch_size = patch_size,
        num_classes = num_classes,
        dim = dim,
        depth = depth,
        heads = heads,
        mlp_dim = mlp_dim,
        dropout = dropout,
        emb_dropout = emb_dropout
        ).cuda()

img = torch.randn(batch_size, channel, image_size, image_size).cuda()

print(v)
print(v.to_patch_embedding(img).shape)

if flag == 0:
        with profile(activities=[torch.profiler.ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_flops=True) as prof:
            with record_function("model_inference"):
                for i in range(iter1):
                    v(img)
        for i in range(len(prof.key_averages())):
            print(prof.key_averages()[i])
        print(prof.key_averages().table(sort_by='cuda_time_total'))

if flag == 1:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for i in range(iter2):
            v(img)

        end.record()
        torch.cuda.synchronize()
        print(start.elapsed_time(end))

