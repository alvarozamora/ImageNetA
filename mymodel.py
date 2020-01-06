import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


class ParallelBranches(nn.Module):
	def __init__(self, classes):
		super().__init__()

		self.classes = classes
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		def Branch():
			branch = nn.Sequential(
			nn.Linear(8192,16), nn.ReLU(inplace=True), nn.BatchNorm1d(16), nn.Dropout(0.5),
			nn.Linear(16,8), nn.ReLU(inplace=True), nn.BatchNorm1d(8), nn.Dropout(0.25),
			nn.Linear(8,1))

			return branch

		self.branches = nn.ModuleList([Branch() for k in range(self.classes)])

	def forward(self, x):
		y = torch.Tensor([]).to(self.device)

		for branch in self.branches:
			y = torch.cat((y, branch(x)), dim = 1)

		return y

class Model(nn.Module):
	def __init__(self, classes):
		super().__init__()

		self.classes = classes

		self.FeatureExtractor = nn.Sequential(
			nn.Conv2d(   3,   128, 6, 2), nn.ReLU(inplace=True), nn.BatchNorm2d(128),
			nn.Conv2d( 128,   256, 6, 2), nn.ReLU(inplace=True), nn.BatchNorm2d(256),
			nn.Conv2d( 256,   256, 5, 2), nn.ReLU(inplace=True), nn.BatchNorm2d(256),
			nn.Conv2d( 256,   512, 5, 2), nn.ReLU(inplace=True), nn.BatchNorm2d(512),
			nn.Conv2d( 512,  512, 5, 2), nn.ReLU(inplace=True), nn.BatchNorm2d(512))


		self.Tree = ParallelBranches(self.classes)

	def forward(self, x):

		# Apply Convolutional Feature Extractor
		x = self.FeatureExtractor(x)
		x = torch.flatten(x, 1)


		x = self.Tree(x)

		return x
