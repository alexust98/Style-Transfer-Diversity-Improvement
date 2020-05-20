from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
from torchvision import transforms
from skimage.color import rgb2hsv, hsv2rgb

from skimage.io import imsave

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from time import time
import argparse

from tqdm import tqdm

def normalize_batch(batch):
    # normalization before input to VGG, using imagenet mean and std. Input should be inside [0,1] interval.
    mean = batch.new_tensor([0.40760392, 0.45795686, 0.48501961]).view(1,-1, 1, 1)  # new tensor with the same dtyp`e and device as batch
    return (batch[:, [2, 1, 0], :, :] - mean)*255.0

def denormalize_batch(batch):
    mean = batch.new_tensor([0.40760392, 0.45795686, 0.48501961]).view(1,-1, 1, 1)  # new tensor with the same dtyp`e and device as batch
    return (batch[:, [2, 1, 0], :, :]/255.0 + mean[:, [2, 1, 0], :, :])


#vgg definition that conveniently let's you grab the outputs from any layer
class VGG_Gatys(nn.Module):
    def __init__(self, pool='max'):
        super(VGG_Gatys, self).__init__()
        #vgg modules
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_4 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        if pool == 'max':
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif pool == 'avg':
            self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool3 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool4 = nn.AvgPool2d(kernel_size=2, stride=2)
            self.pool5 = nn.AvgPool2d(kernel_size=2, stride=2)
            
    def forward(self, x, out_keys):
        out = {}
        out['r11'] = F.relu(self.conv1_1(x))
        out['r12'] = F.relu(self.conv1_2(out['r11']))
        out['p1'] = self.pool1(out['r12'])
        out['r21'] = F.relu(self.conv2_1(out['p1']))
        out['r22'] = F.relu(self.conv2_2(out['r21']))
        out['p2'] = self.pool2(out['r22'])
        out['r31'] = F.relu(self.conv3_1(out['p2']))
        out['r32'] = F.relu(self.conv3_2(out['r31']))
        out['r33'] = F.relu(self.conv3_3(out['r32']))
        out['r34'] = F.relu(self.conv3_4(out['r33']))
        out['p3'] = self.pool3(out['r34'])
        out['r41'] = F.relu(self.conv4_1(out['p3']))
        out['r42'] = F.relu(self.conv4_2(out['r41']))
        out['r43'] = F.relu(self.conv4_3(out['r42']))
        out['r44'] = F.relu(self.conv4_4(out['r43']))
        out['p4'] = self.pool4(out['r44'])
        out['r51'] = F.relu(self.conv5_1(out['p4']))
        out['r52'] = F.relu(self.conv5_2(out['r51']))
        out['r53'] = F.relu(self.conv5_3(out['r52']))
        out['r54'] = F.relu(self.conv5_4(out['r53']))
        out['p5'] = self.pool5(out['r54'])
        return [out[key] for key in out_keys]

def get_input_optimizer(input_img, lr=1.):
    # this line to show that input is a parameter that requires a gradient
    optimizer = optim.LBFGS([input_img.requires_grad_()], lr=lr)
    return optimizer


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        return F.mse_loss(input, self.target)

    
class GramMatrixLossRandomProjection(nn.Module):
    def __init__(self, target_feature, n_proj):
        super(GramMatrixLossRandomProjection, self).__init__()
        b_size, ch, h, w = target_feature.size()
        
        self.n_proj = max(1, int(n_proj * ch))
        
        self.proj = torch.randn(b_size, self.n_proj, ch, device=target_feature.device)
        self.proj = self.proj / torch.norm(self.proj, dim=1, keepdim=True)
        
        self.target = self.gram_matrix(target_feature).detach()

    def gram_matrix(self, y): # returns matrix of cross products on layer y, divided by (height*weight*#channels)
        # YOUR CODE HERE
        b_size, ch, h, w = y.size()
        proj = self.proj
            
        y = torch.bmm(proj, y.view(b_size, ch, h * w))
        return torch.bmm(y, y.transpose(1, 2)) / (h * w * self.n_proj)
        
    def forward(self, input):
        return F.mse_loss(self.gram_matrix(input), self.target)
    
    def __str__(self):
        return 'GramMatrixLossRandomProjection'
    
    
class GramMatrixLossRandomAxis(nn.Module):
    def __init__(self, target_feature, n_axis):
        super(GramMatrixLossRandomAxis, self).__init__()
        b_size, ch, h, w = target_feature.size()
        
        self.n_axis = max(1, int(n_axis * ch))
       
        self.proj = torch.zeros(b_size, self.n_axis, ch, device=target_feature.device)
        for batch_num in range(b_size):
            mask = (torch.rand(self.n_axis)*ch).long().to(target_feature.device)
            self.proj[batch_num, np.arange(self.n_axis), mask] = np.random.choice([1.0, -1.0])
            
        self.target = self.gram_matrix(target_feature).detach()

    def gram_matrix(self, y): # returns matrix of cross products on layer y, divided by (height*weight*#channels)
        # YOUR CODE HERE
        b_size, ch, h, w = y.size()
        proj = self.proj
            
        y = torch.bmm(proj, y.view(b_size, ch, h * w))
        return torch.bmm(y, y.transpose(1, 2)) / (h * w * self.n_axis)
        
    def forward(self, input):
        return F.mse_loss(self.gram_matrix(input), self.target)
    
    def __str__(self):
        return 'GramMatrixLossRandomAxis'
        
        
def run_style_transfer(vgg, style_loss_module, content_img, style_img, input_img, num_steps,
                       style_weight, verbose, lr, style_layers=['r11','r21','r31','r41', 'r51'], content_layers=['r42'], **kwargs):
    """Run the style transfer."""
    if verbose:
        print('Building the style transfer model..')
    optimizer = get_input_optimizer(input_img, lr=lr)
    
    style_losses = []
    content_losses = []
    
    for style_feature in vgg(normalize_batch(style_img), style_layers):
        style_losses.append(style_loss_module(style_feature, **kwargs))
        
    for content_feature in vgg(normalize_batch(content_img), content_layers):
        content_losses.append(ContentLoss(content_feature))
    
    total_time = [0.]
    start_time = time()
    run = [0]
    pbar = tqdm(total=num_steps, desc="Optimizing..")
    while run[0] < num_steps:

        def closure():
            # correct the values of updated input image
            st = time()

            optimizer.zero_grad()
            
            outputs = vgg(input_img, style_layers + content_layers)
            
            style_score = 0.
            content_score = 0.
            
            for i in range(len(style_layers)):
                style_score += style_losses[i](outputs[i])
                
            for j in range(len(style_layers), len(style_layers + content_layers)):
                content_score += content_losses[j - len(style_layers)](outputs[j])

            style_score *= style_weight

            loss = style_score + content_score
            loss.backward()

            total_time[0] += time() - st
            run[0] += 1
            pbar.update(1)
            if (run[0] % 50 == 0):
                if verbose:
                    print("Iteration num. {}:".format(run))
                    print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                        style_score.item(), content_score.item()))
                    print()
                    #imshow(input_img)

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    input_img = denormalize_batch(input_img).clamp_(0.0, 1.0)
    t = time() - start_time
    if verbose:
        print('Execution time: {}'.format(t))
    pbar.close()
    return input_img
    

def image_loader(image_name):
    image = Image.open(image_name)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)


def imshow(tensor, title=None):
    image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = unloader(image)
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.001) # pause a bit so that plots are updated
    

def _imsave(tensor, name, title=None):
    image = tensor.clone().detach().cpu()  # we clone the tensor to not do changes on it
    image = image.squeeze(0)      # remove the fake batch dimension
    image = 255*image.transpose(0, 1).transpose(1, 2)
    imsave(name, image.type(torch.uint8))
    
def parse_arguments():
    parser = argparse.ArgumentParser(description='Real-time style transfer with diversification',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--content', default='content.jpg', help='Content image to be stylized')
    parser.add_argument('--style', default='style.jpg', help='Style image')
    parser.add_argument('--out_dir', default='./', help='Directory where stylized images will be stored')
    parser.add_argument('--style_weight', default=1e3, type=float, help='Non-negative float parameter, controlling stylization strength')
    parser.add_argument('--output', default="Result.jpg", help='Stylized image name')
    parser.add_argument('--n_proj', default=0.6, type=float, help='Portion of random projections')
    parser.add_argument('--iter_num', default=1000, type=int, help='Number of iterations')
    parser.add_argument('--batch_size', default=4, type=int, help='Number of generated images')
    args = parser.parse_args()
    return args


if __name__ == '__main__':

    args = vars(parse_arguments())
    print("\n======Model Parameters======")
    for item in args.items():
        key, value = item
        print(key, ": ", value)
    print("============================\n")
    
    transform = transforms.Compose([
    transforms.ToTensor(),
    #     transforms.Lambda(lambda x: x.mul(255))
    ])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    #get network
    vgg_gatys = VGG_Gatys()
    vgg_gatys.load_state_dict(torch.load('gatysmodel/PytorchNeuralStyleTransfer/vgg_conv.pth'))
    for param in vgg_gatys.parameters():
        param.requires_grad = False
    if torch.cuda.is_available():
        vgg_gatys.cuda()

    imsize = (512, 512) if torch.cuda.is_available() else (128, 128)  # use small size if no gpu

    loader = transforms.Compose([
        transforms.Resize(imsize),  # scale imported image
        transforms.ToTensor()])  # transform it into a torch tensor

    unloader = transforms.ToPILImage()  # reconvert into PIL image
    
    style_img = image_loader(args["style"])
    content_img = image_loader(args["content"])
    
    content_imgs = content_img.clone()
    style_imgs = style_img.clone()
    for i in range(1, args["batch_size"]):
        content_imgs = torch.cat((content_imgs, content_img), dim=0)
        style_imgs = torch.cat((style_imgs, style_img), dim=0)
    
    input_imgs = content_imgs.clone()
    for i in range(content_imgs.size()[0]):
        img = np.array(content_imgs[i].clone().detach().cpu().transpose(0, 1).transpose(1, 2))
        img_hsv = rgb2hsv(img)
        img_hsv[:, :, 0] = np.fmod(img_hsv[:, :, 0] * np.random.rand(1) + np.random.rand(1), 1.0)
        input_imgs[i] = torch.tensor(hsv2rgb(img_hsv)).transpose(2, 1).transpose(1, 0).cuda()
    input_imgs = normalize_batch(input_imgs)
    
    style_params = {"n_proj" : args["n_proj"]}
    iter_num = args["iter_num"]
    
    output = run_style_transfer(vgg_gatys, GramMatrixLossRandomProjection, content_imgs.clone().detach(), style_imgs.clone().detach(), input_imgs.clone().detach(), 
            style_weight=args["style_weight"], num_steps=args["iter_num"], verbose=False, 
            lr=1.0, **style_params)
    for i in range(args["batch_size"]):
        _imsave(output[i].unsqueeze(0), args["out_dir"] + str(i+1)+ "_" + args["output"])