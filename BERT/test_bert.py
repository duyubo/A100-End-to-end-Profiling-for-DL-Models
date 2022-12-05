import torch
import numpy as np
from torch import nn
import torch.autograd.profiler as profiler
from torch.profiler import profile, record_function, ProfilerActivity
from transformers import BertModel, BertConfig

from transformers import BertTokenizer, BertModel
import torch

"""***************** configuration here *******************"""

flag = 0 # 0: bert profiling, 1: bert end-to-end run time 
iter1 = 1000
iter2 = 3000
batch_size = 6

"""***************** configuration end *******************"""


tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
model = BertModel.from_pretrained("bert-large-uncased").cuda()

document = ''
with open('./file.txt') as f:
    document = document.join(f.readlines())
document = document.split(' ')

sentences = []
mystring = ''
for j in range(6):
    for i in document[j * 510 :(j + 1) * 510]:
        mystring += ' ' + i 
    sentences.append(mystring)

print('batch size: ', len(sentences))
inputs = tokenizer(sentences , padding=True, return_tensors="pt", truncation=True)

#new_inputs = torch.rand(6, 512, 1024)
#model.encoder.layer[0](new_inputs.cuda())

# warm up
model(inputs['input_ids'].cuda(), 
                    inputs['token_type_ids'].cuda(), 
                    inputs['attention_mask'].cuda())
if flag == 0:
    with profile(activities=[torch.profiler.ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_flops=True) as prof:
        with record_function("model_inference"):
            for i in range(iter1):
                model(inputs['input_ids'].cuda(), 
                    inputs['token_type_ids'].cuda(), 
                    inputs['attention_mask'].cuda())
                #model.encoder.layer[0](new_inputs.cuda())
    for i in range(len(prof.key_averages())):
        print(prof.key_averages()[i])
    print(prof.key_averages().table(sort_by='cuda_time_total'))

if flag == 1:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for i in range(iter2):
        model(inputs['input_ids'].cuda(),
            inputs['token_type_ids'].cuda(),
            inputs['attention_mask'].cuda())
        #model.encoder.layer[0](new_inputs.cuda())

    end.record()
    torch.cuda.synchronize()
    print(start.elapsed_time(end))

