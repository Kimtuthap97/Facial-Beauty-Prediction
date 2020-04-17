import net
import torch
import time
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms
import numpy as np
import PIL

def ToTensor(pic):
    img = torch.from_numpy(np.array(pic, np.int16, copy=False))
    nchannel = 3
    img = img.view(pic.size[1], pic.size[0], nchannel)
    img = img.transpose(0, 1).transpose(0, 2).contiguous()
    if isinstance(img, torch.ByteTensor):
        return img.float()
    else:
        return img
def de_norm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

def to_var(x, requires_grad=True):
    if not requires_grad:
        return Variable(x, requires_grad=requires_grad)
    else:
        return Variable(x)
"""
    path: path to pretrained model
    img_A: path to image A
    img_B: path to image B
    
    output: fake_A (256x256x3)
"""
def test(img_A, img_B, path='./28_1260_G.pth'):
    start = time.time()
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    transform_mask = transforms.Compose([transforms.Resize(256, interpolation=PIL.Image.NEAREST),ToTensor])
    img_A = transform(Image.open(img_A))
    img_B = transform(Image.open(img_B))
    # Load trained parameters
    model = net.Generator_branch(64, 6)
    
    model.load_state_dict(torch.load(path))
    model.eval()
    
    img_A = img_A[None, :, :, :]
    img_B = img_B[None, :, :, :]
    real_org = to_var(img_A)
    real_ref = to_var(img_B)

    # Get makeup result
    fake_A, fake_B = model(real_org, real_ref)
    rec_B, rec_A = model(fake_B, fake_A)
    result=np.zeros((256, 256, 3))
    result[:, :, 0]=de_norm(fake_A.detach()[0]).numpy()[0]
    result[:, :, 1]=de_norm(fake_A.detach()[0]).numpy()[1]
    result[:, :, 2]=de_norm(fake_A.detach()[0]).numpy()[2]
    print('Done in {}'.format(time.time()-start))
    return result