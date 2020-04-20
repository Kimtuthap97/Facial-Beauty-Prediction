import api.net as net
import torch
import time
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms
import numpy as np
import PIL
import os
import glob
from os.path import join, dirname, realpath
import cv2
import random
makeup_path = './makeup_images'

FOLDER = join(dirname(realpath(__file__)))

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
def test(img_A, path='248_2520_G.pth'):
    start = time.time()
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    transform_mask = transforms.Compose([transforms.Resize(256, interpolation=PIL.Image.NEAREST),ToTensor])
    img_A = transform(Image.open(os.path.join(FOLDER, img_A)))

    list_makeup = glob.glob(os.path.join(FOLDER, makeup_path, '*.png'))
    print(os.path.join(FOLDER, makeup_path, '*.png'))
    img_B = random.choice(list_makeup)
    img_B = transform(Image.open(os.path.join(FOLDER, img_B)))
    model = net.Generator_branch(64, 6)
    if torch.cuda.is_available():
        model.load_state_dict(torch.load(os.path.join(FOLDER, path)))
    else:
        model.load_state_dict(torch.load(os.path.join(FOLDER, path), map_location=lambda storage, loc: storage))
    model.eval()
    img_A = img_A[None, :, :, :]
    img_B = img_B[None, :, :, :]
    real_org = to_var(img_A)
    real_ref = to_var(img_B)

    # Get makeup result
    fake_A, fake_B = model(real_org, real_ref)
    # rec_B, rec_A = model(fake_B, fake_A)
    del model
    result=np.zeros((256, 256, 3))
    result[:, :, 0]=de_norm(fake_A.detach()[0]).numpy()[0]
    result[:, :, 1]=de_norm(fake_A.detach()[0]).numpy()[1]
    result[:, :, 2]=de_norm(fake_A.detach()[0]).numpy()[2]
    result = cv2.resize(result, (350, 350))
    result = cv2.normalize(result, None, alpha = 0, beta = 255, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_32F)
    result = result.astype(np.uint8)
    duration = round(time.time()-start, 3)
    # print('Done in {0} s'.format(duration))
    return result, duration

if __name__ == "__main__":
    test('./vFG383.png')
