import torch
from torch import nn
import torchvision
from torch.nn import functional as F
from efficientnet_pytorch import EfficientNet
import timm



class Detector(nn.Module):

    def __init__(self):
        super(Detector, self).__init__()
        # self.net=EfficientNet.from_pretrained("efficientnet-b4",advprop=True,num_classes=2)
        self.net = timm.create_model("legacy_xception.tf_in1k", pretrained=True, num_classes=2)


        self.cel=nn.CrossEntropyLoss()
        
        

    def forward(self,x):
        x = (x + 1) / 2

        x=self.net(x)
        return x