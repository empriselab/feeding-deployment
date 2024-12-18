import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import torchvision.models as models
from .resnet_dilated_multi import Resnet34_8s

class MiniSPANet(nn.Module):
        def __init__(self, out_features=4, img_height=200, img_width=200):
                super(MiniSPANet, self).__init__()
                self.img_height = img_height
                self.img_width = img_width
                self.resnet = Resnet34_8s(num_classes=1, out_features=out_features)
                self.sigmoid = torch.nn.Sigmoid()
        def forward(self, x):
                heatmap, cls = self.resnet(x) 
                heatmaps = self.sigmoid(heatmap[:,:1, :, :])
                return heatmaps, cls

if __name__ == '__main__':
	model = MiniSPANet()
	x = torch.rand((1,3,200,200))
	heatmap = model.forward(x)
	heatmap, linear_output = model.forward(x)
	rot  = linear_output[:,0]
	cls = linear_output[:,1:]
	print(heatmap.shape, rot.shape, cls.shape)
