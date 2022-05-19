import os
os.chdir('/mnt/kb')
CODE_DIR = 'encoder4editing'
os.chdir(f'./{CODE_DIR}')

from argparse import Namespace
import time
import sys
import numpy as np
from PIL import Image
import gc
import torch
import torchvision.transforms as transforms

sys.path.append(".")
sys.path.append("..")

from utils.common import tensor2im
from models.psp import pSp  # we use the pSp framework to load the e4e encoder.

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

gc.collect()
torch.cuda.empty_cache()

experiment_type = 'ffhq_encode'

EXPERIMENT_DATA_ARGS = {
    "ffhq_encode": {
        "model_path": "/mnt/kb/encoder4editing/pretrained_models/e4e_ffhq_encode.pt"
    }    
}
# Setup required image transformations
EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[experiment_type]
if experiment_type == 'ffhq_encode':
    EXPERIMENT_ARGS['transform'] = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    resize_dims = (256, 256)

model_path = EXPERIMENT_ARGS['model_path']
ckpt = torch.load(model_path, map_location='cpu')
opts = ckpt['opts']
opts['checkpoint_path'] = model_path
opts= Namespace(**opts)
net = pSp(opts)
net.eval()
net.cuda()
print('Model successfully loaded!')

image_path = '/mnt/kb/sanghyun.jpg'

original_image = Image.open(image_path)
original_image = original_image.convert("RGB")

def run_alignment(image_path):
  import dlib
  from utils.alignment import align_face
  predictor = dlib.shape_predictor("/mnt/kb/shape_predictor_68_face_landmarks.dat")
  aligned_image = align_face(filepath=image_path, predictor=predictor) 
  print("Aligned image has shape: {}".format(aligned_image.size))
  return aligned_image 

if experiment_type == "ffhq_encode":
  input_image = run_alignment(image_path)
else:
  input_image = original_image

input_image.resize(resize_dims)
img_transforms = EXPERIMENT_ARGS['transform']
transformed_image = img_transforms(input_image)
def run_on_batch(inputs, net):
    images, latents = net(inputs.to("cuda").float(), randomize_noise=False, return_latents=True)
    return images, latents
with torch.no_grad():
    tic = time.time()
    images, latents = run_on_batch(transformed_image.unsqueeze(0), net)
    result_image, latent = images[0], latents[0]
    toc = time.time()
    print('Inference took {:.4f} seconds.'.format(toc - tic))

torch.save(latents, '/mnt/kb/latents.pt')

os.chdir('/mnt/kb/StyleCLIP/global/')

arr = ['w', 's', 's_mean_std']
for i in arr:
    os.system('python GetCode.py --code_type {} ' .format(i))

import tensorflow as tf
import clip
import pickle
import copy
from MapTS import GetFs,GetBoundary,GetDt
from manipulate import Manipulator

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device) 

M=Manipulator(dataset_name='ffhq') 
fs3=np.load('./npy/ffhq/fs3.npy')
np.set_printoptions(suppress=True)

image_path = None
img_index = 0

latents=torch.load('/mnt/kb/latents.pt') # located in the top-level folder in the Colab UI
w_plus=latents.cpu().detach().numpy()
dlatents_loaded=M.W2S(w_plus)

img_indexs=[img_index]
dlatent_tmp=[tmp[img_indexs] for tmp in dlatents_loaded]
M.num_images=len(img_indexs)

M.alpha=[0]
M.manipulate_layers=[0]
codes,out=M.EditOneC(0, dlatent_tmp) 
original=Image.fromarray(out[0,0]).resize((512,512))
M.manipulate_layers=None

neutral='face with hair'
target='face with blond hair'
classnames=[target,neutral]
dt=GetDt(classnames,model)

beta = 0.2
alpha = 5.5
M.alpha=[alpha]
boundary_tmp2,c=GetBoundary(fs3,dt,M,threshold=beta)
codes=M.MSCode(dlatent_tmp,boundary_tmp2)
out=M.GenerateImg(codes)
generated=Image.fromarray(out[0,0])#.resize((512,512))

resize_image = generated.resize((512, 512))
resize_image.save('/mnt/kb/sanghyun_1.jpg')
