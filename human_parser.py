from __future__ import print_function
import argparse
from datetime import datetime
import os
import sys
import time
import scipy.misc
import scipy.io as sio
import cv2
from glob import glob
import imutils
from matplotlib import pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"]="-1"


import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from PIL import Image
from human_parsing2.utils import *

N_CLASSES = 20
DATA_DIR = './human_parsing2/datasets/CIHP'
LIST_PATH = './human_parsing2/datasets/CIHP/list/val.txt'
DATA_ID_LIST = './human_parsing2/datasets/CIHP/list/val_id.txt'
with open(DATA_ID_LIST, 'r') as f:
    NUM_STEPS = len(f.readlines()) 
RESTORE_FROM = './human_parsing2/checkpoint'

"""Create the model and start the evaluation process."""

# Create queue coordinator.
#coord = tf.train.Coordinator()
# Load reader.
with tf.name_scope("create_inputs"):
    input_img = tf.placeholder(tf.float32, shape=[None,None,3])
    reader = ImageReader(DATA_DIR, LIST_PATH, DATA_ID_LIST, None, False, False, False, input_img)
    image = reader.image

    image_rev = tf.reverse(image, tf.stack([1]))
    image_list = reader.image_list

image_batch = tf.stack([image, image_rev])

h_orig, w_orig = tf.to_float(tf.shape(image_batch)[1]), tf.to_float(tf.shape(image_batch)[2])
image_batch050 = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 0.50)), tf.to_int32(tf.multiply(w_orig, 0.50))]))
image_batch075 = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 0.75)), tf.to_int32(tf.multiply(w_orig, 0.75))]))
image_batch125 = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 1.25)), tf.to_int32(tf.multiply(w_orig, 1.25))]))
image_batch150 = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 1.50)), tf.to_int32(tf.multiply(w_orig, 1.50))]))
image_batch175 = tf.image.resize_images(image_batch, tf.stack([tf.to_int32(tf.multiply(h_orig, 1.75)), tf.to_int32(tf.multiply(w_orig, 1.75))]))


# Create network.
with tf.variable_scope('', reuse=False):
    net_100 = PGNModel({'data': image_batch}, is_training=False, n_classes=N_CLASSES)
with tf.variable_scope('', reuse=True):
    net_050 = PGNModel({'data': image_batch050}, is_training=False, n_classes=N_CLASSES)
with tf.variable_scope('', reuse=True):
    net_075 = PGNModel({'data': image_batch075}, is_training=False, n_classes=N_CLASSES)
with tf.variable_scope('', reuse=True):
    net_125 = PGNModel({'data': image_batch125}, is_training=False, n_classes=N_CLASSES)
with tf.variable_scope('', reuse=True):
    net_150 = PGNModel({'data': image_batch150}, is_training=False, n_classes=N_CLASSES)
with tf.variable_scope('', reuse=True):
    net_175 = PGNModel({'data': image_batch175}, is_training=False, n_classes=N_CLASSES)
# parsing net

parsing_out1_050 = net_050.layers['parsing_fc']
parsing_out1_075 = net_075.layers['parsing_fc']
parsing_out1_100 = net_100.layers['parsing_fc']
parsing_out1_125 = net_125.layers['parsing_fc']
parsing_out1_150 = net_150.layers['parsing_fc']
parsing_out1_175 = net_175.layers['parsing_fc']

parsing_out2_050 = net_050.layers['parsing_rf_fc']
parsing_out2_075 = net_075.layers['parsing_rf_fc']
parsing_out2_100 = net_100.layers['parsing_rf_fc']
parsing_out2_125 = net_125.layers['parsing_rf_fc']
parsing_out2_150 = net_150.layers['parsing_rf_fc']
parsing_out2_175 = net_175.layers['parsing_rf_fc']

# edge net
edge_out2_100 = net_100.layers['edge_rf_fc']
edge_out2_125 = net_125.layers['edge_rf_fc']
edge_out2_150 = net_150.layers['edge_rf_fc']
edge_out2_175 = net_175.layers['edge_rf_fc']


# combine resize
parsing_out1 = tf.reduce_mean(tf.stack([tf.image.resize_images(parsing_out1_050, tf.shape(image_batch)[1:3,]),
                                        tf.image.resize_images(parsing_out1_075, tf.shape(image_batch)[1:3,]),
                                        tf.image.resize_images(parsing_out1_100, tf.shape(image_batch)[1:3,]),
                                        tf.image.resize_images(parsing_out1_125, tf.shape(image_batch)[1:3,]),
                                        tf.image.resize_images(parsing_out1_150, tf.shape(image_batch)[1:3,]),
                                        tf.image.resize_images(parsing_out1_175, tf.shape(image_batch)[1:3,])]), axis=0)

parsing_out2 = tf.reduce_mean(tf.stack([tf.image.resize_images(parsing_out2_050, tf.shape(image_batch)[1:3,]),
                                        tf.image.resize_images(parsing_out2_075, tf.shape(image_batch)[1:3,]),
                                        tf.image.resize_images(parsing_out2_100, tf.shape(image_batch)[1:3,]),
                                        tf.image.resize_images(parsing_out2_125, tf.shape(image_batch)[1:3,]),
                                        tf.image.resize_images(parsing_out2_150, tf.shape(image_batch)[1:3,]),
                                        tf.image.resize_images(parsing_out2_175, tf.shape(image_batch)[1:3,])]), axis=0)


edge_out2_100 = tf.image.resize_images(edge_out2_100, tf.shape(image_batch)[1:3,])
edge_out2_125 = tf.image.resize_images(edge_out2_125, tf.shape(image_batch)[1:3,])
edge_out2_150 = tf.image.resize_images(edge_out2_150, tf.shape(image_batch)[1:3,])
edge_out2_175 = tf.image.resize_images(edge_out2_175, tf.shape(image_batch)[1:3,])
edge_out2 = tf.reduce_mean(tf.stack([edge_out2_100, edge_out2_125, edge_out2_150, edge_out2_175]), axis=0)

raw_output = tf.reduce_mean(tf.stack([parsing_out1, parsing_out2]), axis=0)
head_output, tail_output = tf.unstack(raw_output, num=2, axis=0)
tail_list = tf.unstack(tail_output, num=20, axis=2)
tail_list_rev = [None] * 20
for xx in range(14):
    tail_list_rev[xx] = tail_list[xx]
tail_list_rev[14] = tail_list[15]
tail_list_rev[15] = tail_list[14]
tail_list_rev[16] = tail_list[17]
tail_list_rev[17] = tail_list[16]
tail_list_rev[18] = tail_list[19]
tail_list_rev[19] = tail_list[18]
tail_output_rev = tf.stack(tail_list_rev, axis=2)
tail_output_rev = tf.reverse(tail_output_rev, tf.stack([1]))

raw_output_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
raw_output_all = tf.expand_dims(raw_output_all, dim=0)
pred_scores = tf.reduce_max(raw_output_all, axis=3)
raw_output_all = tf.argmax(raw_output_all, axis=3)
pred_all = tf.expand_dims(raw_output_all, dim=3) # Create 4-d tensor.


raw_edge = tf.reduce_mean(tf.stack([edge_out2]), axis=0)
head_output, tail_output = tf.unstack(raw_edge, num=2, axis=0)
tail_output_rev = tf.reverse(tail_output, tf.stack([1]))
raw_edge_all = tf.reduce_mean(tf.stack([head_output, tail_output_rev]), axis=0)
raw_edge_all = tf.expand_dims(raw_edge_all, dim=0)
pred_edge = tf.sigmoid(raw_edge_all)
res_edge = tf.cast(tf.greater(pred_edge, 0.5), tf.int32)




# Which variables to load.
restore_var = tf.global_variables()
# Set up tf session and initialize variables. 
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
init = tf.global_variables_initializer()

sess.run(init)
sess.run(tf.local_variables_initializer())

# Load weights.
loader = tf.train.Saver(var_list=restore_var)
if RESTORE_FROM is not None:
    if load(loader, sess, RESTORE_FROM):
        print(" [*] Load SUCCESS")
    else:
        print(" [!] Load failed...")

# Start queue threads.
#threads = tf.train.start_queue_runners(coord=coord, sess=sess)

def parse(img):
    """
    if img.shape[0]>540:
        img = imutils.resize(img, height=540)
    if img.shape[1]>540:
        img = imutils.resize(img, width=540)
    """
    print("reached 1")
    parsing_= sess.run(pred_all, feed_dict={input_img: img})
    print("reached 2")
    parsing_.shape = parsing_.shape[1:3]
    plt.imshow(parsing_)
    #msk = decode_labels(parsing_, num_classes=N_CLASSES)
    #coord.request_stop()
    #coord.join(threads)
    return parsing_
