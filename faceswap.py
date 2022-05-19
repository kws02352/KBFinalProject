import os
os.chdir('/mnt/kb')

import cv2
import numpy as np
from FaceSwap import face_main
from dlib import get_frontal_face_detector as face_detector
from PIL import Image, ImageEnhance

def swap(source_img,target_img,result_img):

    image_info = {}

    image_info['src'] = source_img
    image_info['dst'] = target_img
    image_info['out'] = result_img

    face_main.faceswap(image_info)

    return result_img

def faceswapMain(source_path, target_path):    
    source_img = '/mnt' + source_path
    target_img = '/mnt' + target_path
    result_img = '/mnt' + target_path
    swap(source_img, target_img, result_img)

if __name__ == "__main__":
    None