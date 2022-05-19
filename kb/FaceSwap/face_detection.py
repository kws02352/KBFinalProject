import cv2
import os
import dlib
import numpy as np

def face_detection(img,upsample_times=1):
    detector = dlib.get_frontal_face_detector()
    faces = detector(img, upsample_times)

    return faces

ROOT_PATH = os.getcwd() + u'/FaceSwap'
PREDICTOR_PATH = 'models/shape_predictor_68_face_landmarks.dat'
PREDICTOR_PATH = ROOT_PATH + '/' + PREDICTOR_PATH
predictor = dlib.shape_predictor(PREDICTOR_PATH)

# Face and points detection
def face_points_detection(img, bbox:dlib.rectangle):
    # Get the landmarks/parts for the face in box d.
    shape = predictor(img, bbox)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    coords = np.asarray(list([p.x, p.y] for p in shape.parts()), dtype=np.int)

    # return the array of (x, y)-coordinates
    return coords

def select_face(im, r=10):
    faces = face_detection(im)

    if len(faces) == 1:
        idx = np.argmax([(face.right() - face.left()) * (face.bottom() - face.top()) for face in faces])
        bbox = faces[idx]

    points = np.asarray(face_points_detection(im, bbox))

    im_w, im_h = im.shape[:2]
    left, top = np.min(points, 0)
    right, bottom = np.max(points, 0)

    x, y = max(0, left - r), max(0, top - r)
    w, h = min(right + r, im_h) - x, min(bottom + r, im_w) - y

    return points - np.asarray([[x, y]]), (x, y, w, h), im[y:y + h, x:x + w]
