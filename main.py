# Aims

# Run a segmentation model
# Identify segmentation masks of drivable lanes
import json
import pprint

import json
import vis
import matplotlib.pyplot as plt
import sys

# temp work around for using open cv
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import images

def getBbox(bbox):
    return [int(bbox['x1']), int(bbox['y1']), int(bbox['x2'] - bbox['x1']),  int(bbox['y2'] - bbox['y1'])]


BASE_PATH = './bdd100k'
labels = '/labels'
imagesPath = '/images/100k/train/'
# train_file = '/bdd100k_labels_images_train.json'
train_file = '/train_sample.json'
with open(BASE_PATH + labels + train_file) as f:
    data = json.load(f)
    d = data[0]
    # print(d)
    img_name = d['name']
    bbox = getBbox(d['labels'][0]['box2d'])
    catName = d['labels'][0]['category']
    print(bbox)
    path_to_image_file = BASE_PATH + imagesPath + img_name

    img = images.retrieveImage(path_to_image_file)

    fig,ax = plt.subplots(1)
    plt.axis('off')
    plt.imshow(img)
    vis.plotBoundingBoxWithText(plt,ax,bbox,catName)
    plt.show()

