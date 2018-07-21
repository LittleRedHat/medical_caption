import torch
import torch.nn as nn
import torchvision.models as M 



class ResnetExtractor(nn.Module):
    def __init__(self,resnet,layers):
        super(ResnetExtractor,self).__init__()
        self.resnet = resnet
        ## layers means which bottleneck
        self.resnet_layers = [len(self.resnet.layer1),len(self.resnet.layer2),len(self.resnet.layer3),len(self.resnet.layer4)]
        self.layers = layers

    def forward(self,x):
        print(self.resnet_layers)
        output = []
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        
        for i,block in enumerate(self.resnet.layer1.children()):
            x = block(x)
            
            if i in self.layers:
                output.append(x)

        for i,block in enumerate(self.resnet.layer2.children()):
            x = block(x)
            if (i + sum(self.resnet_layers[:1])) in self.layers:
                output.append(x)

        for i,block in enumerate(self.resnet.layer3.children()):
            x = block(x)
            if (i + sum(self.resnet_layers[:2])) in self.layers:
                output.append(x)

        for i,block in enumerate(self.resnet.layer4.children()):
            x = block(x)
            if (i + sum(self.resnet_layers[:2])) in self.layers:
                output.append(x)

        x = self.resnet.avgpool(x)
        x = x.view(x.size(0), -1)
        return x,output



resnet = M.resnet50()

model = ResnetExtractor(resnet,[0,5])

x = torch.tensor(torch.zeros((1,3,224,224)),dtype=torch.float)

model(x)        
        



