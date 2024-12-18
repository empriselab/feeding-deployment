import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
import torchvision.models as models
from .resnet_dilated import Resnet34_8s

class RecenterNet(nn.Module):
        def __init__(self, num_keypoints=1, img_height=136, img_width=136):
                super(RecenterNet, self).__init__()
                self.img_height = img_height
                self.img_width = img_width
                self.resnet = Resnet34_8s()
                self.num_keypoints = num_keypoints
                self.sigmoid = torch.nn.Sigmoid()
        def forward(self, x):
                heatmap = self.resnet(x) 
                heatmaps = self.sigmoid(heatmap[:,:self.num_keypoints, :, :])
                return heatmaps

if __name__ == '__main__':
	model = RecenterNet()
	x = torch.rand((1,3,200,200))
	heatmap = model.forward(x)
	heatmap, linear_output = model.forward(x)
	print(x.shape, heatmap.shape)
