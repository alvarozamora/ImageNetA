import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from mymodel import Model
from tqdm import tqdm
from LSUV import LSUVinit
from torchsummary import summary


load = False
rf = 0 if not load else 10
epochs = 10
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# My Modifications to the Model
class fc(nn.Module):
	def __init__(self):
		super().__init__()



if __name__ == '__main__':
	# Load the Model
	#model = torchvision.models.resnext101_32x8d().to(device)
	model = Model(200).to(device)
	model.device = device
	summary(model, (3,224,224))
	if load == False:
		model.init = False
		model.iter = 0


	# Load the Data
	transforms = torchvision.transforms.Compose([
		torchvision.transforms.RandomSizedCrop(224),
		torchvision.transforms.ToTensor(),
		torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
							std=[0.229, 0.224, 0.225])
		])
	dataset = torchvision.datasets.ImageFolder("imagenet-a", transform = transforms)
	trainloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

	#Optimizer
	optimizer = torch.optim.Adam(model.parameters(), lr = 3e-4)

	# Start Train Loop
	losses = []
	for epoch in range(rf, epochs):
		N = 0
		#avg_acc = 0
		loader = tqdm(trainloader)
		for q, (I, label) in enumerate(loader, 0):
			# Send to device
			I, label = I.to(device), label.to(device)

			# Layer Sequential Unit Variance (LSUV) Initialization
			if epoch == 0 and q == 0 and load == False and model.init == False:
				#model = LSUVinit(model, I, std_tol= 1e-3, cuda=True)
				model.init = True

			# Forward Pass
			out = model.forward(I)
			model.iter += 1

			# Compute Loss and Derivatives
			loss = F.cross_entropy(out, label)
			loss.backward()

			# Compute Accuracy
			preds = out.argmax(axis=1)
			if q > 0:
				avg_acc = avg_acc*N
			else:
				avg_acc = 0
			N += I.size(0)
			acc = (preds == label).float()
			avg_acc += acc.sum()
			avg_acc /= N
			acc = acc.mean()

			# Show Status
			loader.set_description(
				(
					f'epoch: {epoch + 1}; batch {q:03d} loss: {loss.item():.5f}; acc: {acc.item()*100:.5f}; avg_acc = {avg_acc.item()*100:.5f}'
				)
				)
			#print("Epoch", epoch, "batch", q, "loss", np.round(loss.item(),4), "acc", acc)


			# Save Loss
			if model.iter%100 == 0:
				losses.append([loss.item()])

			# Step
			optimizer.step()
			optimizer.zero_grad()

		# Save Model, Optimizer, and Loss
		torch.save(model, f'ckpts/Model{epoch:04d}.pth')
		torch.save(optimizer.state_dict(), f'ckpts/Optim{epoch:04d}.pth')
		np.save('Loss.npy', np.array(losses))
