# Aims

# Run a segmentation model
# Identify segmentation masks of drivable lanes
import json
import pprint

import json
import vis
import matplotlib.pyplot as plt
import sys
import pickle
# temp work around for using open cv
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')

import images

def getBbox(bbox):
    return [int(bbox['x1']), int(bbox['y1']), int(bbox['x2'] - bbox['x1']),  int(bbox['y2'] - bbox['y1'])]


BASE_PATH = './bdd100k'
imagesPath = '/images/100k/train/'
trafficLightsData = pickle.load(open('trafficLights.pkl','rb'))

for i,d in enumerate(trafficLightsData):
    if(i%1000 == 0):
        print(i)
    # print(d)
    img_name = d['name']
    path_to_image_file = BASE_PATH + imagesPath + img_name
    img = images.retrieveImage(path_to_image_file)
    fig,ax = plt.subplots(1)
    plt.axis('off')
    plt.imshow(img)

    for index in range(len(d['labels'])):
        try:
            bbox = getBbox(d['labels'][index]['box2d'])
            catName = d['labels'][index]['category']

            if catName == 'traffic light':
                print(bbox)
                path_to_image_file = BASE_PATH + imagesPath + img_name

                img = images.retrieveImage(path_to_image_file)
                vis.plotBoundingBoxWithText(plt,ax,bbox,catName)
        except:
            pass

    plt.show()

print("Total Length of traffic lights list: ", len(trafficLightImages))



