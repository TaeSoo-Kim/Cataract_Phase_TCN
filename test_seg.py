import numpy as np
import cv2
import os
import pdb
import matplotlib.pyplot as plt
import lmdb 
import Models
from keras import backend as K
from keras.applications.resnet50 import ResNet50
import sys
from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint


colors = ['w','r','c','m','y','g','b','r','c','m','y']
classes = {
0:"Background",
1:"Side_Incision",
2:"Main_Incision",
3:"Capsulorhexis",
4:"Hydrodissection",
5:"Phacoemulsification",
6:"Cortical_Removal",
7:"Lens_Insertion",
8:"OVD_Removal",
9:"Corneal_Hydration",
10:"Suture_Incision"
}
test_lengths = {
  "114" : 15421,
  "008" : 29637,
  "014" : 76659,
  "015" : 80957,
  "023" : 51890,
  "026" : 168711,
  "038" : 61598,
  "039" : 100155,
  "052" : 54350,
  "053" : 156930,
  "055" : 144359
}
video_dir = "/home/tk/dev/data/CataractMount/Pilot2015/processed/videos/procedures/"
data_out_dir = '/media/tk/EE44DA8044DA4B4B/cataract_phase_img'
weights_path = '/home/tk/dev/tksrc/cataract_phase/weights/ResNet50Pretrain_phase_ep:029_acc:0.020_loss:0.990.h5'
annotations_dir = "/home/tk/dev/data/CataractMount/Pilot2015/processed/annotations/tasks/lfang6/"
height = 224
width = 224
skip_rate = 1
batch = 32
nb_classes = 11

def seg_with_spatial_net():
  current_batch_count = 1

  activation = "relu"                                                           ## CHECK THIS!!!!!!!!!                                           ## CHECK THIS!!!!!!!!!
  momentum = 0.9
  lr = 0.01
  optimizer = SGD(lr=lr, momentum=momentum, decay=0.0, nesterov=True)           ## CHECK THIS!!!!!!!!!
  loss = 'categorical_crossentropy'
  model = Models.resnet(nb_classes)
  model.compile(optimizer,
                loss=loss, 
                metrics=['accuracy'])
  model.load_weights(weights_path)

  for vid_num in test_lengths.keys():
    video_path = os.path.join(video_dir,'vid_'+vid_num+'.avi')
    sample_num = 0
    cap = cv2.VideoCapture(video_path)
    total_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    #video_y = np.zeros((total_num,nb_classes))
    video_y = np.array([0]*total_num)

    num_correct = 0
    y_list = np.array([0]*total_num)
    #set up label
    y_file = open(os.path.join(annotations_dir,"vid_"+vid_num+"_tasks.txt"),'r')
    for line in y_file:
      sp = line.split()
      y_list[int(sp[0]):int(sp[1])] = int(sp[2])

    X = np.zeros((batch,height,width,3))
    for frame_num in range(0,total_num,skip_rate):
    #for frame_num in range(0,100,skip_rate):
      cap.set(cv2.CAP_PROP_POS_FRAMES,frame_num)
      ret,img = cap.read()
      if not ret:
        break
      img = cv2.resize(img,(width,height))
      X[frame_num%batch] = (img.astype(np.float) - 128)/128.0

      if (frame_num+1) % batch == 0:
        y = model.predict_on_batch(X)
        y_argmax = np.array([np.argmax(yy) for yy in y])
        y_argmax_true = y_list[frame_num-batch+1:frame_num+1]
        num_correct += len(np.where(y_argmax_true == y_argmax)[0])
        video_y[frame_num-batch+1:frame_num+1] = y_argmax
        X = np.zeros((batch,height,width,3))
        print("\rProgress:{:06d}/{:06d}, Acc:{:.03f}".format(frame_num,total_num,(num_correct*1.0/(frame_num+1))),end='\r')
        #pdb.set_trace()
      
      
      #pdb.set_trace()
    #pdb.set_trace()
    print("\nVideo: "+vid_num+", Acc: {:.03f}\n".format((num_correct*1.0/(frame_num+1))))
    plot_segmentation(y_list,video_y,vid_name=vid_num, acc=(num_correct*1.0/(frame_num+1)))
    

def plot_segmentation(truth_y, y,vid_name,acc):
  
  ax = plt.subplot(2,1,1)
  ax.set_title("Ground Truth")
  ax.imshow(truth_y[:,None].T, interpolation='nearest',vmin=0, vmax=nb_classes, cmap='Paired')
  plt.yticks([])
  plt.xticks([])
  ax.axis('tight')
  ax = plt.subplot(2,1,2)
  ax.set_title("Video: "+vid_name+", frame wise Acc:{:.03f}".format(acc))
  ax.imshow(y[:,None].T, interpolation='nearest',vmin=0, vmax=nb_classes,cmap='Paired')
  plt.yticks([])
  ax.axis('tight')
  #plt.show()
  plt.savefig(vid_name+".png")
  plt.close('all')

if __name__ == "__main__":
  seg_with_spatial_net()