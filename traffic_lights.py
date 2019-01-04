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
import math
# temp work around for using open cv
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
sys.path.insert(0,'/home/karan/objectdetection/models/research')

import images
import contextlib2
from object_detection.dataset_tools import tf_record_creation_util
import tensorflow as tf
from object_detection.utils import dataset_util
import cv2

def getBbox(bbox):
    return [int(bbox['x1']), int(bbox['y1']), int(bbox['x2'] - bbox['x1']),  int(bbox['y2'] - bbox['y1'])]



BASE_PATH = './bdd100k'
imagesPath = '/images/100k/val/'
trafficLightsData = pickle.load(open('trafficLightsVal.pkl','rb'))
totalSizeOfDataset = len(trafficLightsData)


def create_tf_example(example):

    height = example.height
    width = example.width 
    filename = example.filename 
    encoded_image_data = example.encoded_image_data 
    image_format = example.image_format 

    xmins = example.xmins 
    xmaxs = example.xmaxs 
    ymins = example.ymins 
    ymaxs = example.ymaxs
    classes_text = example.classes_text
    classes = example.classes

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


class Example:

    def __init__(self,img_data):
        self.height = img_data['height']
        self.width = img_data['width'] # Image width
        self.filename = img_data['name'].encode() # Filename of the image. Empty if image is not from file
        self.encoded_image_data = img_data['encodedImg'] # Encoded image bytes
        self.image_format = b'jpeg' # b'jpeg' or b'png'

        self.xmins = img_data['xmins'] # List of normalized left x coordinates in bounding box (1 per box)
        self.xmaxs = img_data['xmaxs'] # List of normalized right x coordinates in bounding box
                # (1 per box)
        self.ymins = img_data['ymins']# List of normalized top y coordinates in bounding box (1 per box)
        self.ymaxs = img_data['ymaxs'] # List of normalized bottom y coordinates in bounding box
                # (1 per box)
        self.classes_text = img_data['classes_text'] # List of string class name of bounding box (1 per box)
        self.classes = img_data['classes'] # List of integer class id of bounding box (1 per box)



def getDatasetExamplesByBatch(index, batch_size):

    examples = []
    start = index*batch_size
    end = start + batch_size
    print(start, end)
    for i,d in enumerate(trafficLightsData[start:end]):
        if(i%100 == 0):
            print(i)

        img_name = d['name']
        path_to_image_file = BASE_PATH + imagesPath + img_name
        img = images.retrieveImage(path_to_image_file)

        xmins = []
        xmaxs = []
        ymins = []
        ymaxs = []
        classes_text = []
        classes = []

        path_to_image_file = BASE_PATH + imagesPath + img_name
        img = images.retrieveImage(path_to_image_file)
        d['height'] = img.shape[0]
        d['width'] = img.shape[1]
        success, encoded_img = cv2.imencode('.jpeg', img)

        d['encodedImg'] = encoded_img.tobytes()

        for index in range(len(d['labels'])):

            catName = d['labels'][index]['category']
            if catName == 'drivable area':
                continue

            if catName == 'traffic light':
                # bbox = getBbox(d['labels'][index]['box2d'])
                xmins.append(d['labels'][index]['box2d']['x1']/d['width'])
                ymins.append(d['labels'][index]['box2d']['y1']/d['height'])
                xmaxs.append(d['labels'][index]['box2d']['x2']/d['width'])
                ymaxs.append(d['labels'][index]['box2d']['y2']/d['height'])
                classes_text.append(catName.encode())
                classes.append(1)

        d['xmins'] = xmins 
        d['xmaxs'] = xmaxs 
        d['ymins'] = ymins
        d['ymaxs'] = ymaxs
        d['classes'] = classes
        d['classes_text'] = classes_text

        examples.append(Example(d))

    print("Total Length of traffic lights list: ", len(examples))
    return examples


def main(_):

    batch_size = 500
    num_shards = math.ceil(totalSizeOfDataset/batch_size)
    index = 0
    output_filebase='./trafficLightsDataset/val/traffic_lights_val_dataset.record'
    print('Writing to TF File')
    
    print(totalSizeOfDataset)
    print(num_shards)

    with contextlib2.ExitStack() as tf_record_close_stack:
        output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(
            tf_record_close_stack, output_filebase, num_shards)
        while index < num_shards:
            examples = getDatasetExamplesByBatch(index, batch_size)
            index += 1
            for i, example in enumerate(examples):
                tf_example = create_tf_example(example)
                output_shard_index = (batch_size*index + i) % num_shards
                output_tfrecords[output_shard_index].write(tf_example.SerializeToString())

if __name__ == '__main__':
    tf.app.run()



