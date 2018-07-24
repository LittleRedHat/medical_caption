import torch
import torch.nn as nn
import torchvision.models as M 

import numpy as np
import json

# vgg = M.vgg19()
# modules = vgg.features.children()
# x = torch.zeros((1,3,224,224),dtype=torch.float)
# layers = [34,20]
            
# for index,module in enumerate(modules):
    
#     print('**********')
#     print(index)
#     print(module)
#     x = module(x)


# pipe = nn.Sequential(*(list(vgg.features.children())[34:36]))
# print(pipe)




# finding_file = '../output/preprocess/IU_Chest_XRay/findings.json'
# with open(finding_file,'r') as f:
#     findings = json.load(f)

# tags_file = '../output/preprocess/IU_Chest_XRay/tags.csv'




# lstm = nn.LSTM(128,512,num_layers=1,batch_first=True,bidirectional=True)

# a = torch.zeros((32,128))
# a = a.unsqueeze(1)

# output,hidden = lstm(a)
# print(hidden.size())
# print(output.size())

vgg = M.vgg19_bn(pretrained=True)
modules = vgg.features.children()
x = torch.zeros((1,3,224,224),dtype=torch.float)
for index,module in enumerate(modules):
    print('**********')
    print(index)
    print(module)
    x = module(x)
    print(x.size())


    

