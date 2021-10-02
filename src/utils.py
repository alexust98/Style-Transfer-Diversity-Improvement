import torch
from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt
from skimage.io import imsave


def normalize_batch(batch):
	# normalization before input to VGG, using imagenet mean and std. Input should be inside [0,1] interval.
	mean = batch.new_tensor([0.40760392, 0.45795686, 0.48501961]).view(1,-1, 1, 1)  # new tensor with the same dtyp`e and device as batch
	return (batch[:, [2, 1, 0], :, :] - mean)*255.0

def denormalize_batch(batch):
	mean = batch.new_tensor([0.40760392, 0.45795686, 0.48501961]).view(1,-1, 1, 1)  # new tensor with the same dtyp`e and device as batch
	return (batch[:, [2, 1, 0], :, :]/255.0 + mean[:, [2, 1, 0], :, :])

def image_loader(image_name, imsize, device):
	image = Image.open(image_name)
	# fake batch dimension required to fit network's input dimensions
	loader = transforms.Compose([
		transforms.Resize(imsize),  # scale imported image
		transforms.ToTensor()
	])  # transform it into a torch tensor
	
	image = loader(image).unsqueeze(0)
	return image.to(device, torch.float)

def imshow(tensor, title=None):
	image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
	image = image.squeeze(0)	  # remove the fake batch dimension
	
	unloader = transforms.ToPILImage()  # reconvert into PIL image
	image = unloader(image)
	plt.imshow(image)
	if title is not None:
		plt.title(title)
	plt.pause(0.001) # pause a bit so that plots are updated

def _imsave(tensor, name, title=None):
	image = tensor.clone().detach().cpu()  # we clone the tensor to not do changes on it
	image = image.squeeze(0)	  # remove the fake batch dimension
	image = 255*image.transpose(0, 1).transpose(1, 2)
	imsave(name, image.type(torch.uint8))