import numpy as np
import pdb
import os
import lmdb
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Dropout,
    Flatten
)
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.regularizers import l2
from keras import backend as K
from keras.optimizers import RMSprop,SGD,Adam
from keras.callbacks import ModelCheckpoint

import Models

sample_lengths = {
  "051": 72430,
  "034": 64023,
  "037" : 50420,
  "138" : 57358,
  "148" : 72943,
  "146" : 58774,
  "154" : 64787,
  "133" : 103201,
  "149" : 72720,
  "181" : 72227,
  "085" : 178894,
  "123" : 77667,
  "128" : 46666,
  "058" : 144699,
  "121" : 124511,
  "158" : 57779,
  "161" : 72678,
  "147" : 96183,
  "120" : 60826,
  "173" : 112929,
  "157" : 65450,
  "127" : 46149,
  "118" : 73080,
  "175" : 100128,
  "130" : 44896,
  "117" : 55253,
  "062" : 131539,
  "145" : 144017,
  "169" : 69739,
  "119" : 91751,
  "137" : 87483,
  "042" : 107666,
  "122" : 56280,
  "164" : 40380,
  "170" : 85719,
  "162" : 69459,
  "077" : 184373,
  "155" : 47399,
  "150" : 54106,
  "124" : 35481
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

def train():
  data_out_dir = '/media/tk/EE44DA8044DA4B4B/cataract_phase_img'
  height = 224
  width = 224
  skip_rate = 10
  batch = 32
  nb_classes = 11
  nb_epoch = 100
  current_batch_count = 1

  out_dir_name = 'ResNet50Pretrain_phase'                             ## CHECK THIS!!!!!!!!!
  activation = "relu"                                                           ## CHECK THIS!!!!!!!!!
  momentum = 0.9
  lr = 0.01
  optimizer = SGD(lr=lr, momentum=momentum, decay=0.0, nesterov=True)           ## CHECK THIS!!!!!!!!!
  loss = 'categorical_crossentropy'
  model = Models.resnet(nb_classes)
  model.compile(optimizer,
                loss=loss, 
                metrics=['accuracy'])

  X = np.zeros((batch,height,width,3))
  Y = np.zeros((batch,nb_classes))

  for e in range(0,nb_epoch):
    ACC = 0.
    LOSS = 0.
    N = 0
    for vid_num in sample_lengths.keys():
      lmdb_env_x = lmdb.open(os.path.join(data_out_dir,vid_num+"X"))
      lmdb_txn_x = lmdb_env_x.begin()
      lmdb_cursor_x = lmdb_txn_x.cursor()

      lmdb_env_y = lmdb.open(os.path.join(data_out_dir,vid_num+"y"))
      lmdb_txn_y = lmdb_env_y.begin()
      lmdb_cursor_y = lmdb_txn_y.cursor()

      
      indices = list(range(0,int(sample_lengths[vid_num]/skip_rate)))
      np.random.shuffle(indices)

      label = np.frombuffer(lmdb_cursor_y.get('{:0>8d}'.format(0).encode()),dtype=np.dtype(np.int64))
      for index in indices:
        real_frame_ind = index*skip_rate
        try:
          value = np.frombuffer(lmdb_cursor_x.get('{:0>8d}'.format(index).encode()),dtype=np.dtype(np.uint8))
        except:
          continue
          #pdb.set_trace()

        x = value.reshape((height,width,3))
        x.setflags(write=1)
        x = x.astype(np.float)
        x -= 128
        x /= 128.0
        y = label[real_frame_ind]

        X[current_batch_count] = x
        Y[current_batch_count,y] = 1
        current_batch_count += 1

        if (current_batch_count % batch) == 0:
          losses = model.train_on_batch(X, Y)
          ACC += losses[1]  # current accuracy distinguishing real-vs-fake
          LOSS += losses[0]
          N += 1

          print("epoch: {:03d} | loss: {:.03f} | acc: {:.03f} \r".format(e,LOSS/N,ACC/N), end='\r')
          ## TRAIN()
          
          X = np.zeros((batch,height,width,3))
          Y = np.zeros((batch,nb_classes))
          current_batch_count = 0
    print("Finished with epoch:", e,"\n")
    model_file = './weights/'+ out_dir_name + '_ep:%03d_acc:%0.3f_loss:%0.3f.h5' % (e+1,(LOSS/N),(ACC/N))
    model.save_weights(model_file, overwrite=True)
    #test_loss = model.test_on_batch()

def test():
  data_out_dir = '/media/tk/EE44DA8044DA4B4B/cataract_phase_img'
  weights_path = '/home/tk/dev/tksrc/cataract_phase/weights/ResNet50Pretrain_phase_ep:029_acc:0.020_loss:0.990.h5'
  height = 224
  width = 224
  skip_rate = 10
  batch = 32
  nb_classes = 11
  nb_epoch = 100
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

  X = np.zeros((batch,height,width,3))
  Y = np.zeros((batch,nb_classes))
  for vid_num in test_lengths.keys():
    ACC = 0.
    LOSS = 0.
    N = 0
    for vid_num in sample_lengths.keys():
      lmdb_env_x = lmdb.open(os.path.join(data_out_dir,vid_num+"X"))
      lmdb_txn_x = lmdb_env_x.begin()
      lmdb_cursor_x = lmdb_txn_x.cursor()

      lmdb_env_y = lmdb.open(os.path.join(data_out_dir,vid_num+"y"))
      lmdb_txn_y = lmdb_env_y.begin()
      lmdb_cursor_y = lmdb_txn_y.cursor()

      indices = list(range(0,int(sample_lengths[vid_num]/skip_rate)))
      np.random.shuffle(indices)

      label = np.frombuffer(lmdb_cursor_y.get('{:0>8d}'.format(0).encode()),dtype=np.dtype(np.int64))
      for index in indices:
        real_frame_ind = index*skip_rate
        try:
          value = np.frombuffer(lmdb_cursor_x.get('{:0>8d}'.format(index).encode()),dtype=np.dtype(np.uint8))
        except:
          continue
          #pdb.set_trace()

        x = value.reshape((height,width,3))
        x.setflags(write=1)
        x = x.astype(np.float)
        x -= 128
        x /= 128.0
        y = label[real_frame_ind]

        X[current_batch_count] = x
        Y[current_batch_count,y] = 1
        current_batch_count += 1

        if (current_batch_count % batch) == 0:
          losses = model.test_on_batch(X, Y)
          ACC += losses[1]  # current accuracy distinguishing real-vs-fake
          LOSS += losses[0]
          N += 1

          print("loss: {:.03f} | acc: {:.03f} \r".format(LOSS/N,ACC/N), end='\r')
          ## TRAIN()
          
          X = np.zeros((batch,height,width,3))
          Y = np.zeros((batch,nb_classes))
          current_batch_count = 0
    print("Finished testing")
    print("loss: {:.03f} | acc: {:.03f} \r".format(LOSS/N,ACC/N), end='\n')


def lengths():
  video_dir = "/home/tk/dev/data/CataractMount/Pilot2015/processed/videos/procedures/"
  annotations_dir = "/home/tk/dev/data/CataractMount/Pilot2015/processed/annotations/tasks/lfang6/"
  data_out_dir = '/media/tk/EE44DA8044DA4B4B/cataract_phase_img'
  files = os.listdir(annotations_dir)
  for f in files:
    dot_ind = f.rfind('.')
    if f[dot_ind:] == '.txt':
      vid_num = f[4:7]
      video_path = os.path.join(video_dir,'vid_'+vid_num+'.avi')
      sample_num = 0
      cap = cv2.VideoCapture(video_path)
      total_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
      print(vid_num,":",str(total_num)+',')
      cap.release()

if __name__ == "__main__":
  #lengths()
  #train()
  test()