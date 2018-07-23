import torch
import torch.nn as nn
import torchvision.models as M 

import numpy as np
import json

vgg = M.vgg19()
modules = vgg.features.children()
x = torch.zeros((1,3,224,224),dtype=torch.float)
layers = [34,20]
            
for index,module in enumerate(modules):
    
    print('**********')
    print(index)
    # print(module)
    x = module(x)
    
    if index in layers:
        print(x.size())

# finding_file = '../output/preprocess/IU_Chest_XRay/findings.json'
# with open(finding_file,'r') as f:
#     findings = json.load(f)

# tags_file = '../output/preprocess/IU_Chest_XRay/tags.csv'




