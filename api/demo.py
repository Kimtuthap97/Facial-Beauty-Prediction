import api.net as net
import torch
import time
from torch.autograd import Variable
from PIL import Image
from torchvision import transforms
import numpy as np
import PIL
import os 
from os.path import join, dirname, realpath
from mtcnn import MTCNN
import cv2

FOLDER = join(dirname(realpath(__file__)))

def crop_face(img):
    img = cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB)
    # print(img)
    detector = MTCNN()
    
#     print('img.shape', img.shape)
    faces=detector.detect_faces(img)
#     print(faces)
    if len(faces)==0:
        print('{0}: No face found')
        return img[0:1, 0:1, 0:1]
    else:
        faces=faces[0]['box']
        x, y, z, k = faces[0], faces[1], faces[2], faces[3]
#         print('x, y, z, k', x, y, z, k)
        ext = [z, k][np.argmax([z, k])]
        ext=int(ext*1.2)
        x=int(x-0.5*int(ext-z))
        if x < 0:
            x =0
        if y < 0:
            y=0
#         plt.imshow(temp[y:y+ext, x:x+ext, :])
        print('Done cropping faces')
        # print(cv2.cvtColor(img[y:y+ext, x:x+ext, :], cv2.COLOR_RGB2BGR))
        return img[y:y+ext, x:x+ext, :]

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
def test(img_A, img_B='vFG840.png', path='152_1260_G.pth'):
    start = time.time()
    transform = transforms.Compose([transforms.Resize(256),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    transform_mask = transforms.Compose([transforms.Resize(256, interpolation=PIL.Image.NEAREST),ToTensor])
    img_A = cv2.resize(crop_face(img_A), (256, 256))
    # print(img_A)
    img_A = Image.fromarray(img_A)
    # img_A = img_A.resize((256, 256))
    img_A = transform(img_A)
    img_B = transform(Image.open(os.path.join(FOLDER, img_B)))
    # Load trained parameters

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
    result=np.zeros((256, 256, 3))
    result[:, :, 0]=de_norm(fake_A.detach()[0]).numpy()[0]
    result[:, :, 1]=de_norm(fake_A.detach()[0]).numpy()[1]
    result[:, :, 2]=de_norm(fake_A.detach()[0]).numpy()[2]
    duration = round(time.time()-start, 3)
    print('Done in {0} s'.format(duration))
    return result, duration

if __name__ == "__main__":
    test('./vFG383.png')
