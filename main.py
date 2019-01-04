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
labels = '/labels'
imagesPath = '/images/100k/train/'
train_file = '/bdd100k_labels_images_val.json'
# train_file = '/train_sample.json'
trafficLightImages = []
# train_file = '/train_sample.json'
with open(BASE_PATH + labels + train_file) as f:
    data = json.load(f)
    print(len(data))
    for i,d in enumerate(data):
        if(i%1000 == 0):
            print(i)
        # print(d)
        # img_name = d['name']
        for index in range(len(d['labels'])):
            # bbox = getBbox(d['labels'][0]['box2d'])
            catName = d['labels'][index]['category']
            if catName == 'traffic light':
                trafficLightImages.append(d)
                break
            # print(bbox)
            # path_to_image_file = BASE_PATH + imagesPath + img_name

            # img = images.retrieveImage(path_to_image_file)

            # fig,ax = plt.subplots(1)
            # plt.axis('off')
            # plt.imshow(img)
            # vis.plotBoundingBoxWithText(plt,ax,bbox,catName)
            # plt.show()
print("Total Length of traffic lights list: ", len(trafficLightImages))

with open('trafficLightsVal.pkl','wb') as f:
    print('pickling')
    pickle.dump(trafficLightImages,f)
