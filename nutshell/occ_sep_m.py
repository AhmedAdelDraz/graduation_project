from __future__ import division
import tensorflow as tf
import tensornets as nets
import numpy as np
import matplotlib.pyplot as plt
import math
from IPython.display import clear_output
import random
import cv2
from copy import copy, deepcopy
from pathlib import Path
import os
import time 
from datetime import timedelta
from tqdm import tqdm
#import zipfile
import tarfile
import shutil
import wget
import sys
import voc
from utils import *
from messages import send_message



C1=[238, 72, 58, 24, 203, 230, 54, 167, 246, 136, 106, 95, 226, 171, 43, 159, 231, 101, 65, 157]
C2=[122, 71, 173, 32, 147, 241, 53, 197, 228, 164, 4, 209, 175, 223, 176, 182, 48, 3, 70, 13]
C3=[148, 69, 133, 41, 157, 137, 125, 245, 89, 85, 162, 43, 16, 178, 197, 150, 13, 140, 177, 224]
idx_to_labels=['aeroplane','bicycle','bird','boat','bottle','bus','car','cat','chair','cow','diningtable','dog','horse','motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
def visualize_img(img,bboxes,thickness,name):
  img=img.reshape(img.shape[1],img.shape[1],3)
  for c, boxes_c in enumerate(bboxes):
    for b in boxes_c:
      #ul_x, ul_y=b[0]-b[2]/2.0, b[1]-b[3]/2.0
      #br_x, br_y=b[0]+b[2]/2.0, b[1]+b[3]/2.0

      #ul_x, ul_y=(min(max(int(ul_x),0),415),min(max(int(ul_y),0),415))
      #br_x, br_y=(min(max(int(br_x),0),415),min(max(int(br_y),0),415))

      ul_x, ul_y=int(b[0]), int(b[1])
      br_x, br_y=int(b[2]), int(b[3])

      color_class=(C1[c], C2[c], C3[c])
      img=cv2.rectangle(img, (ul_x, ul_y), (br_x, br_y), color=color_class, thickness=3) 
      label = '%s: %.2f' % (idx_to_labels[c], b[-1]) 
      labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1) 
      ul_y = max(ul_y, labelSize[1]) 
      img=cv2.rectangle(img, (ul_x, ul_y - labelSize[1]), (ul_x + labelSize[0], ul_y + baseLine),color_class, cv2.FILLED) 
      img=cv2.putText(img, label, (ul_x, ul_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0)) 

  cv2.imwrite(name+'.jpg', img)
  #cv2_imshow(img)
  return img

tf.reset_default_graph() # It's importat to resume training from latest checkpoint 

voc_dir = '/home/alex054u4/data/nutshell/newdata/VOCdevkit/VOC%d'

# Define the model hyper parameters
is_training = tf.placeholder(tf.bool)
N_classes=20
x = tf.placeholder(tf.float32, shape=(None, 416, 416, 3), name='input_x')
yolo=model(x)
# Define an optimizer
step = tf.Variable(0, trainable=False)
lr = tf.train.piecewise_constant(
    step, [100, 180, 320, 570, 1000, 10000, 60000],
    [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-4, 1e-5])
train = tf.train.MomentumOptimizer(lr, 0.9).minimize(yolo.loss,global_step=step)

current_epo= tf.Variable(0, name = 'current_epo',trainable=False,dtype=tf.int32)

#Check points
checkpoint_path   = "/home/alex054u3/data/nutshell/training_trial1"
checkpoint_prefix = os.path.join(checkpoint_path,"ckpt")
if not os.path.exists(checkpoint_path):
  os.mkdir(checkpoint_path)

init_op     = tf.global_variables_initializer()
train_saver = tf.train.Saver(max_to_keep=2)

def evaluate_accuracy(data_type='tr'):
  if (data_type  == 'tr'): acc_data  = voc.load(voc_dir % 2007,'trainval',total_num =100)
  elif(data_type == 'te') : acc_data  = voc.load(voc_dir % 2007, 'test', total_num=100)
  
  #print('Train Accuracy: ',voc.evaluate(boxes, voc_dir % 2007, 'trainval'))
  results = []
  idx     = np.random.randint(100)
  for i,(img,_) in enumerate(acc_data):
    acc_outs = sess.run(yolo, {x: yolo.preprocess(img),is_training: False})
    boxes=yolo.get_boxes(acc_outs, img.shape[1:3])
    results.append(boxes)
    if(i == idx):
      img_vis=img
      boxes_vis=boxes
  if (data_type  =='tr'):
    eval_print=voc.evaluate(results, voc_dir % 2007, 'trainval')
  elif (data_type=='te'):
    visualize_img(yolo.preprocess(img_vis)*255,boxes_vis,5,'img')
    eval_print=voc.evaluate(results, voc_dir % 2007, 'test')
  print('\n')
  print(eval_print)
  return eval_print

with tf.Session() as sess:
  ckpt_files = [f for f in os.listdir(checkpoint_path) if os.path.isfile(os.path.join(checkpoint_path, f)) and 'ckpt' in f]
  if (len(ckpt_files)!=0):
    train_saver.restore(sess,checkpoint_prefix)
  else:
    sess.run(init_op)
    sess.run(yolo.stem.pretrained())

  losses = []
  subject= "Trial 1 on server- Occam's Code -Separable"
  emails_list=['ahmedfakhry805@gmail.com', 'ahmadadelattia@gmail.com']
  print('Starting')
  pbar = tqdm(total = 233)
  pbar.update(35) 
  for i in range(233):
    # Iterate on VOC07+12 trainval once
    
    trains = voc.load_train([voc_dir % 2007, voc_dir % 2012],'trainval', batch_size=64)
    
    for (imgs, metas) in trains:
      # `trains` returns None when it covers the full batch once
      if imgs is None:
        train_saver.save(sess,checkpoint_prefix)
        print_out='epoch:'+str(i)+' lr: '+str(sess.run(lr))+ ' loss:'+str(losses[-1])
        eval_train=evaluate_accuracy()
        break
      metas.insert(0, yolo.preprocess(imgs))  # for `inputs`
      metas.append(True)                      # for `is_training`
      outs= sess.run([train, yolo.loss],dict(zip(yolo.inputs, metas)))
      losses.append(outs[-1])
    
    print(print_out)
    eval_test=evaluate_accuracy(data_type='te')
    pbar.update(1)
    print ('==============================================================================================================================================================')
  pbar.close()
    #if i%10==0 or i==232:
      #message=print_out
      #message+='================================='
      #message+= '\n train acc \n' +eval_train
      #message+='================================='
      #message+= '\n test acc \n' +eval_test
      #send_message(emails_list,subject ,message, ['img.jpg'])
