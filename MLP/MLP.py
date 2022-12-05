import torch
import numpy as np
from torch import nn
import torch.autograd.profiler as profiler
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import BertModel, BertConfig

from transformers import BertTokenizer, BertModel
import torch

"""***************** configuration here *******************"""

flag = 0 # 0: MLP profiling, 1: MLP end-to-end run time 
iter1 = 3000
iter2 = 30000
batch_size = 3072
layer_n = 2
hidd_in = 2048
hidd = 4096
hidd_out = 4096

"""***************** configuration end *******************"""


class MLP(torch.nn.Module):
    def __init__(self, layer_number, h_in, h, h_out):
        super(MLP, self).__init__()
        self.fc = nn.Linear(h_in, h)
        layers = []
        for i in range(layer_number):
            layers.append(nn.Linear(h, h))
            layers.append(nn.ReLU())
        self.l = torch.nn.Sequential(*layers)
        self.h_out = nn.Linear(h, h_out)
    def forward(self, x):
        x = self.fc(x)
        x = self.l(x) # instead of Heaviside step fn
        x = self.h_out(x)
        return x

d = torch.rand(batch_size, hidd_in).cuda()
model = MLP(layer_n, hidd_in, hidd, hidd_out).cuda()
print(model)

if flag == 0: 
    with profile(activities=[torch.profiler.ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_flops=True) as prof:
        with record_function("model_inference"):
            for i in range(iter1):
                model(d)

    for i in range(len(prof.key_averages())):
        print(prof.key_averages()[i])
    print(prof.key_averages().table(sort_by='cuda_time_total'))

if flag == 1:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for i in range(iter2):
        model(d)

    end.record()
    torch.cuda.synchronize()
    print(start.elapsed_time(end))

