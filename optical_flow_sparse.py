from skimage import feature
import numpy as np
import math
import random
import cv2
import os
from os import listdir
from os.path import isfile, join
import csv

import matplotlib
#matplotlib.use("TkAgg")
matplotlib.use('macosx')
from matplotlib import pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx
import matplotlib.animation as animation
import matplotlib.patches as patches
from parula import parula_map

DIR = "cut"
FILENAME = "FLFRAME"
AMPRATIO = 30

# load images
data_dir = "%s/data"%DIR
imgs = [cv2.imread(join(data_dir, f))[30:-30, 30:-30] for f in sorted(listdir(data_dir)) if isfile(join(data_dir, f)) and f.find('.') != 0]
#imgs = [cv2.imread(join(data_dir, f)) for f in sorted(listdir(data_dir)) if isfile(join(data_dir, f)) and f.find('.') != 0]
print(imgs[0].shape)
csvlist = list()

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

# Take first frame and find corners in it
old_gray = cv2.cvtColor(imgs[0], cv2.COLOR_BGR2GRAY)

clusters = feature.blob_log(old_gray, min_sigma=2, max_sigma=7, threshold=0.01)
clusters = random.sample(list(clusters), int(len(clusters)/10))
p0 = np.array([np.array([np.array([np.float32(c[1]), np.float32(c[0])])]) for c in clusters])

# Create some random colors
color = np.random.randint(0,255,(len(clusters),3))
#color = np.random.randint(0, 255, (1, 3))

# Draw initial beads position
for i, c in enumerate(clusters):
    blob_img = cv2.circle(imgs[0], (int(c[1]), int(c[0])), 3, color[i].tolist(), 0)
cv2.imwrite("{}/blob.bmp".format(DIR), blob_img)
print("save blob image")
print(len(clusters))

# start optical flow
displacement = list()
for idx in range(1, len(imgs)):
    frame_gray = cv2.cvtColor(imgs[idx], cv2.COLOR_BGR2GRAY)
    if not os.path.exists("{}/gray".format(DIR)):
        os.makedirs("{}/gray".format(DIR))
    cv2.imwrite("{}/gray/{}.bmp".format(DIR, idx), frame_gray);
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    good_new = p1
    good_old = p0
    
    oldlist = list();
    newlist = list();

    for i,(new,old, state) in enumerate(zip(good_new,good_old, st)):
        a,b = new.ravel()
        c,d = old.ravel()
        oldlist.append([idx-1, i, c, d, state[0]])
        if idx == len(imgs)-1:
            newlist.append([idx, i, a, b, state[0]])
        dis = math.sqrt((a-c)**2+(b-d)**2)
        displacement.append(dis)
    csvlist.extend(oldlist)
    if len(newlist):
        csvlist.extend(newlist)

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1,1,2)

# save csv file
with open("{}/bead_pos.csv".format(DIR), "w") as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerows(csvlist)


# Create a mask image for drawing purposes
mask = np.zeros_like(imgs[0])

'''
# Define the codec and create VideoWriter object
w = imgs[0].shape[0]
h = imgs[0].shape[1]
fps = 1.0
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('{}/{}_output.mp4v'.format(DIR, FILENAME),fourcc, fps, (w, h))
'''

# draw animation
fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_axis_off()
plt.close(fig)
ims = []

mask = np.zeros_like(imgs[0])
cNorm  = colors.Normalize(vmin=min(displacement), vmax=max(displacement))
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=parula_map)
arrows = list()

def update(t):
    for a in arrows:
        a.set_visible(False)
    oldlist = [r for r in csvlist if r[0]==t-1]
    newlist = [r for r in csvlist if r[0]==t]
    for i, (old, new) in enumerate(zip(oldlist, newlist)):
        a, b = new[2:4]
        c, d = old[2:4]
        dis = math.sqrt((a-c)**2+(b-d)**2)
        colorVal = scalarMap.to_rgba(dis)
        arrow = patches.Arrow(c, d, (a-c)*AMPRATIO, (b-d)*AMPRATIO, color=colorVal, width=10)
        ax.add_patch(arrow)
        arrows.append(arrow)
    return arrows

ax.imshow(mask)
anim = animation.FuncAnimation(fig, update, frames=np.arange(1, len(imgs)), interval=800)
anim.save("%s/FLoutput.gif" % DIR, writer='imagemagick', dpi=150)
'''
out.release()
cv2.destroyAllWindows()
'''

# show parula map scale
plt.imshow(np.linspace(0, 100, 256)[None, :], aspect='auto',
                cmap=parula_map, extent=[min(displacement), max(displacement), 0, 256])
plt.show()

print((min(displacement), max(displacement)))

#print(np.mean(displacement))
print(len([d for d in displacement if d<=1]))
print(len([d for d in displacement if d>1 and d<=2]))
print(len([d for d in displacement if d>2 and d<=3]))
print(len([d for d in displacement if d>3 and d<=4]))
print(len([d for d in displacement if d>4 and d<=4.36]))

#print(max([d for d in displacement if d < 16]))

#print(min(displacement))
#print(max(displacement))
#plt.hist(displacement, 100)
#plt.show()

with open("{}/bead_pos.csv".format(DIR), "w") as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerows(csvlist)

