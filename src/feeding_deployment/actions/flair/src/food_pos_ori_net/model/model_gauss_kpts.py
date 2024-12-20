import torch
import time
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import time
import sys
from .resnet_dilated import Resnet34_8s, Resnet18_8s

class KeypointsGauss(nn.Module):
	def __init__(self, num_keypoints, img_height=480, img_width=640):
		super(KeypointsGauss, self).__init__()
		self.num_keypoints = num_keypoints
		self.num_outputs = self.num_keypoints
		self.img_height = img_height
		self.img_width = img_width
		#self.resnet = Resnet34_8s(num_classes=num_keypoints)
		self.resnet = Resnet18_8s(num_classes=num_keypoints)
		self.sigmoid = torch.nn.Sigmoid()
	def forward(self, x):
                output = self.resnet(x) 
                heatmaps = self.sigmoid(output)
                return heatmaps

if __name__ == '__main__':
        device = 'cpu'
        model = KeypointsGauss(2, img_height=200, img_width=200).to(device)
        x = torch.rand((1,3,200,200)).to(device)
        start = time.time()
        result = model.forward(x)
        end = time.time()
        print('here', end-start)
        print(x.shape)
        print(result.shape)
