import torch
import torch.nn as nn
import torch.nn.functional as F
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

mse = ((pred-I)**2).sum(-1).sum(-1).sum(-1)
hl = 100*model.HLoss(pred,I)
kl = torch.max(10*torch.ones_like(model.KLoss), model.KLoss)

channels = 3
grad_x_weights = torch.tensor([[1, -1]], dtype=torch.float32).to(model.device)
grad_x_weights = grad_x_weights.expand(channels, 1, 1, 2)
Gx = F.conv2d(I, grad_x_weights, groups=I.shape[1], padding=1)
gx = F.conv2d(pred, grad_x_weights, groups=pred.shape[1], padding=1)
grad_y_weights = torch.tensor([[ 1],[ -1]], dtype=torch.float32).to(model.device)
grad_y_weights = grad_y_weights.expand(channels, 1, 2, 1)
Gy = F.conv2d(I, grad_y_weights, groups=I.shape[1], padding=1)
gy = F.conv2d(pred, grad_y_weights, groups=pred.shape[1], padding=1)


adS = torch.zeros_like(mse)#1.0*(((Gx-gx)**2).sum(-1).sum(-1).sum(-1) + ((Gy-gy)**2).sum(-1).sum(-1).sum(-1))
bowl = (F.relu(I-1) + F.relu(-I)).sum(-1).sum(-1).sum(-1)
loss = (hl + mse + kl + adS + bowl).mean()
print("Epoch", epoch, "batch", q, "loss", np.round(loss.item(),4), "kl", np.round(kl.mean().item(),4), "mse", np.round(mse.mean().item(),4), "img means", "({}, {})".format(np.round(pred.mean().item(),3), np.round(I.mean().item(),3)))
#print("means", model.gaussian_parameters(model.prior)[0].min().item(),  model.gaussian_parameters(model.prior)[0].max().item(), "std", "({} {})".format(np.round(pred.reshape(pred.size(0),pred.size(1),-1).std(dim=-1).mean().item(), 3), np.round(I.reshape(I.size(0), I.size(1),-1).std(dim=-1).mean().item(), 3)), "DSSIM", hl.mean().item(), "grads", adS.mean().item())
print("std", "({} {})".format(np.round(pred.reshape(pred.size(0),pred.size(1),-1).std(dim=-1).mean().item(), 3), np.round(I.reshape(I.size(0), I.size(1),-1).std(dim=-1).mean().item(), 3)), "DSSIM", hl.mean().item(), "grads", adS.mean().item())
if model.iter%100 == 0:
    losses.append([kl.mean().item(), mse.mean().item(), adS.mean().item(), hl.mean().item()])
loss.backward()
optimizer.step()
optimizer.zero_grad()

if (q + epoch)%50 == 1:
    img = torch.cat((pred[:8],I[:8]),dim=0)
    nrow = I.size(0)
    nrow = 8
    torchvision.utils.save_image(img, 'gen.png', nrow=nrow, padding=1)

#if (epoch+1)%2==0 and q == 0:
if model.iter%500 == 0:
    img = torch.cat((pred[:8],I[:8]),dim=0)
    nrow = I.size(0)
    nrow = 8
    torchvision.utils.save_image(img, 'gen_'+str(epoch+1)+"_"+str(model.iter)+'.png', nrow=nrow, padding=1)
    np.save('Loss.npy', np.array(losses))

#if model.iter == 500:
#    torch.save(optimizer.state_dict(), 'Optim_iter_'+str(model.iter)+'.pth')

    #if ((epoch+1)%1)==0:
      #torch.save(model.state_dict(), 'Model_'+str(epoch+1)+'.pth')
    #  torch.save(model, 'Model_all_'+str(epoch+1)+'.pth')
    #  torch.save(optimizer.state_dict(), 'Optim_'+str(epoch+1)+'.pth')
    #  torch.save(model.prior, 'Model_prior_'+str(epoch+1)+'.pth')
  #torch.save(model.state_dict(), 'Model.pth')
  torch.save(model, 'Model_all.pth')

  np.save('Loss.npy', np.array(losses))
