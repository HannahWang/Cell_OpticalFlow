from skimage import feature
import numpy as np
import math
import cv2
from os import listdir
from os.path import isfile, isdir, join
import itertools
import csv

DIR = "FLdata"
FILENAME = "FLFRAME"

# load images
imgs = [cv2.imread(join(DIR, f)) for f in listdir(DIR) if isfile(join(DIR, f)) and f.find('.') != 0]

# Take first frame and find corners in it
old_gray = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY)

hsv = np.zeros_like(imgs[0])
hsv[...,1] = 255

# Define the codec and create VideoWriter object
w = imgs[0].shape[0]
h = imgs[0].shape[1]
fps = 1.0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('video/{}_dense_output.mp4v'.format(FILENAME),fourcc, fps, (w, h))

for idx in range(1, len(imgs)):
    frame = imgs[idx]
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    cv2.imwrite("video/gray/{}.bmp".format(idx), frame_gray);

    flow = cv2. calcOpticalFlowFarneback(old_gray, frame_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])

    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

    cv2.imshow('frame', bgr)
    out.write(bgr)
    
    cv2.imwrite("denseopticalflow/{}.bmp".format(idx), bgr)

    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()

out.release()
cv2.destroyAllWindows()
