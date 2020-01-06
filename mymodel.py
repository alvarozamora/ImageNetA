import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class Model(nn.Module):
	def __init__(self, classes):
		super().__init__()

		self.classes = classes

		self.FeatureExtractor = nn.Sequential(
			nn.Conv2d(   3,   128, 6, 2), nn.ReLU(inplace=True), nn.BatchNorm2d(128),
			nn.Conv2d( 128,   256, 6, 2), nn.ReLU(inplace=True), nn.BatchNorm2d(256),
			nn.Conv2d( 256,   256, 5, 2), nn.ReLU(inplace=True), nn.BatchNorm2d(256),
			nn.Conv2d( 256,   512, 5, 2), nn.ReLU(inplace=True), nn.BatchNorm2d(512),
			nn.Conv2d( 512,  1024, 5, 2), nn.ReLU(inplace=True), nn.BatchNorm2d(1024),
			nn.Conv2d(1024,  1024, 3, 2), nn.ReLU(inplace=True), nn.BatchNorm2d(1024))

		def Branch():
			branch = nn.Sequential(
			nn.Linear(25600,256), nn.ReLU(inplace=True), nn.BatchNorm1d(256),
			nn.Linear(256,64), nn.ReLU(inplace=True), nn.BatchNorm1d(256),
			nn.Linear(64,1))

			return branch

		self.branches = nn.ModuleList[Branch() for k in range(self.classes)]

	def forward(self, x)

		x = self.FeatureExtractor(x)

		y = torch.Tensor([])

		for branch in branches:
			y = torch.cat((y, branch(x)), dim = 0)

		return y
