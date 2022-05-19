import os
os.chdir('/mnt/kb/StyleCLIP/global/')
arr = ['w', 's', 's_mean_std']
for i in arr:
    os.system('python GetCode.py --code_type {} ' .format(i))

from MapTS import GetFs,GetBoundary,GetDt
from manipulate import Manipulator
import tensorflow as tf
from PIL import Image
import numpy as np
import pickle
import torch
import clip
import copy

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_styleclip_model():
    os.chdir('/mnt/kb/StyleCLIP/global/')
    model, preprocess = clip.load("ViT-B/32", device=device)
    print('Model successfully StyleGAN loaded!')
    return model 

def styleImg(model, imgName, styleNum):
    styleKinds = {'1':'face with blonde hair', '2':'face with Quiff Hair', '3':'face with afro hair', '4':'face with Hip-top Fade', '5':'face with bang hair', '6':'face with black hair', '7':'face with curly long hair', '8':'face with curly short hair'}
    styleBetas = {'1':0.2, '2':0.16, '3':0.14, '4':0.15, '5':0.17, '6':0.3, '7':0.16,'8':0.16}
    styleAlphas = {'1':5.5, '2':1.0, '3':5.0, '4':7.0, '5':1.0, '6':2.0, '7':3.0 , '8':3.0}
    os.chdir('/mnt/kb/StyleCLIP/global/')
    
    M=Manipulator(dataset_name='ffhq') 
    fs3=np.load('./npy/ffhq/fs3.npy')
    np.set_printoptions(suppress=True)
    
    image_path = '/mnt/media/'+imgName.split('.')[0]
    img_index = 0
    
    latents=torch.load('/mnt/latentVector/'+imgName.split('.')[0]+'.pt') # located in the top-level folder in the Colab UI
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
    target=styleKinds[styleNum]
    classnames=[target,neutral]
    dt=GetDt(classnames,model)
    
    beta = styleBetas[styleNum]
    alpha = styleAlphas[styleNum]
    M.alpha=[alpha]
    boundary_tmp2,c=GetBoundary(fs3,dt,M,threshold=beta)
    codes=M.MSCode(dlatent_tmp,boundary_tmp2)
    out=M.GenerateImg(codes)
    generated=Image.fromarray(out[0,0])#.resize((512,512))
    
    resize_image = generated.resize((512, 512))
    resize_image.save('/mnt/media/result/'+imgName.split('.')[0]+'.jpg')
    
if __name__=="__main__":
    None