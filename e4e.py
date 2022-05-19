import os
os.chdir('/mnt/kb')
CODE_DIR = 'encoder4editing'
os.chdir(f'./{CODE_DIR}')
from argparse import Namespace
from PIL import Image
import numpy as np
import time
import sys
import gc

import torch
import torchvision.transforms as transforms

sys.path.append(".")
sys.path.append("..")

from utils.common import tensor2im
from models.psp import pSp  # we use the pSp framework to load the e4e encoder.

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

experiment_type = ''
EXPERIMENT_ARGS = {}
resize_dims = ()

def load_e4e_model():
    global experiment_type
    global EXPERIMENT_ARGS
    global resize_dims
    
    gc.collect()
    torch.cuda.empty_cache()
    
    experiment_type = ''
    EXPERIMENT_ARGS = {}
    resize_dims = ()
    
    os.chdir('/mnt/kb')
    CODE_DIR = 'encoder4editing'
    os.chdir(f'./{CODE_DIR}')
    
    experiment_type = 'ffhq_encode'
    
    EXPERIMENT_DATA_ARGS = {
        "ffhq_encode": {
            "model_path": "/mnt/kb/encoder4editing/pretrained_models/e4e_ffhq_encode.pt"
        }    
    }
    # Setup required image transformations
    EXPERIMENT_ARGS = EXPERIMENT_DATA_ARGS[experiment_type]
    
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
    return net
    
def check(imgName):
    latentRoot = '/mnt/latentVector'

    latentList = os.listdir(latentRoot)  
    
    latentName = imgName.split('.')[0] + '.pt'
    
    if latentName in latentList:
        os.remove(os.path.join(latentRoot, latentName))
    
def img2latent(net, imgName):
    os.chdir('/mnt/kb')
    CODE_DIR = 'encoder4editing'
    os.chdir(f'./{CODE_DIR}')
    
    image_path = '/mnt/media/'+imgName

    original_image = Image.open(image_path)
    original_image = original_image.convert("RGB")
    
    def run_alignment(image_path):
      import dlib
      from utils.alignment import align_face
      predictor = dlib.shape_predictor("/mnt/kb/shape_predictor_68_face_landmarks.dat")
      aligned_image = align_face(filepath=image_path, predictor=predictor) 
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
    
    torch.save(latents, '/mnt/latentVector/'+imgName.split('.')[0]+'.pt')
    
    
if __name__=="__main__":
    None