#! /usr/bin/env python
import os
import cv2

from skimage.exposure import match_histograms
from FaceSwap.face_detection import select_face
from FaceSwap.face_swap import face_swap
# 히스토그램 매칭
def histogram_specification(reference, image):
    matched = match_histograms(image, reference, multichannel=True)
    return matched

def faceswap(args):
    src_img = cv2.imread(args['src'])
    dst_img = cv2.imread(args['dst'])

    # src_img = histogram_specification(dst_img, src_img)

    #얼굴 선택
    src_points, src_shape, src_face = select_face(src_img)
    dst_points, dst_shape, dst_face = select_face(dst_img)

    output = face_swap(src_face, dst_face, src_points, dst_points, dst_shape, dst_img, args)

    dir_path = os.path.dirname(args['out'])
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    cv2.imwrite(args['out'], output)
