import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
from LSUV import LSUVinit



load = False

# My Modifications to the Model
class fc(nn.Module):
	def __init__(self):
		super().__init__()



if __name__ == '__main__':
	# Load the Model
	model = torchvision.models.resnext101_32x8d()
	if load == False:
		model.init = False
		model.iter = 0


	# Load the Data
	transforms = torchvision.transforms.Compose([
		torchvision.transforms.ToTensor()
		])
	dataset = torchvision.datasets.ImageFolder("imagenet-a", transforms = transforms)
	loader = torch.utils.DataLoader(dataset, batch_size=32, shuffle=True, sampler=None,
		batch_sampler=None, num_workers=4, collate_fn=None,
		pin_memory=True, drop_last=False, timeout=0,
		worker_init_fn=None)

	# Start Train Loop
	for epoch in range(rf, epochs):
		for q, (data, label) in enumerate(trainloader, 0):
			I, _ = data
			I = I.float().to(device)

			if epoch == 0 and q == 0 and load == False and model.init == False:
				model = LSUVinit(model, I, std_tol= 1e-3, cuda=True)
				model.init = True

			# forward + backward + optimize
			model.iter += 1
			pred = model.forward(I)

			loss = (hl + mse + kl + adS + bowl).mean()
			loss.backward()

			print("Epoch", epoch, "batch", q, "loss", np.round(loss.item(),4), "kl", np.round(kl.mean().item(),4), "mse", np.round(mse.mean().item(),4), "img means", "({}, {})".format(np.round(pred.mean().item(),3), np.round(I.mean().item(),3)))

			if model.iter%100 == 0:
				losses.append([loss.item()])

			optimizer.step()
			optimizer.zero_grad()

		torch.save(model, f'Model{epoch:04d}.pth')
		np.save('Loss.npy', np.array(losses))
