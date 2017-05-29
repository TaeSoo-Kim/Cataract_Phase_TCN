import numpy as np
import cv2
import os
import pdb
import matplotlib.pyplot as plt
import lmdb 
import Models
from keras import backend as K
from keras.applications.resnet50 import ResNet50

from keras.models import Model
from keras.optimizers import RMSprop,SGD,Adam

import Models

import sys

def get_feature(X,model):
  get_3rd_layer_output = K.function([model.layers[0].input, K.learning_phase()],
                                  [model.layers[-2].output])
  return get_3rd_layer_output([X,0])[0]

def write_phase_tcn():
  data_src_dir = '/media/tk/EE44DA8044DA4B4B/cataract_phase_img'
  data_out_dir = '/media/tk/EE44DA8044DA4B4B/cataract_phase_tcn'
  weights_path = '/home/tk/dev/tksrc/cataract_phase/weights/ResNet50Pretrain_phase_ep:029_acc:0.020_loss:0.990.h5'
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
  height = 224
  width = 224
  skip_rate = 10
  batch = 32
  nb_classes = 11
  nb_epoch = 100
  current_batch_count = 0
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
  out_ind = 0
  for vid_num in sample_lengths.keys():
    lmdb_env_x = lmdb.open(os.path.join(data_src_dir,vid_num+"X"))
    lmdb_txn_x = lmdb_env_x.begin()
    lmdb_cursor_x = lmdb_txn_x.cursor()

    OUT_lmdb_file_x = os.path.join(data_src_dir,vid_num+'X')
    OUT_lmdb_env_x = lmdb.open(OUT_lmdb_file_x, map_size=int(1e12))
    OUT_lmdb_txn_x = OUT_lmdb_env_x.begin(write=True)

    
    indices = list(range(0,int(sample_lengths[vid_num]/skip_rate)))

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
        X_feat = get_feature(X,model)
        for x in X_feat:
          keystr = '{:0>8d}'.format(out_ind)
          out_ind += 1
          OUT_lmdb_txn_x.put( keystr.encode(), X.tobytes() )
          OUT_lmdb_txn_x.commit()
        OUT_lmdb_txn_x = OUT_lmdb_env_x.begin(write=True)
        print(vid_num," Written {:06d} / {:06d}\r".format(index,len(indices)), end='\r')

        ## exctract feature
        
        X = np.zeros((batch,height,width,3))
        Y = np.zeros((batch,nb_classes))
        current_batch_count = 0
    print("\n Done with ", vid_num)

def write_phase_image():
  video_dir = "/home/tk/dev/data/CataractMount/Pilot2015/processed/videos/procedures/"
  annotations_dir = "/home/tk/dev/data/CataractMount/Pilot2015/processed/annotations/tasks/lfang6/"
  data_out_dir = '/media/tk/EE44DA8044DA4B4B/cataract_phase_img'
  files = os.listdir(annotations_dir)
  height = 224
  width = 224
  skip_rate = 10
  batch = 128
  for f in files:
    dot_ind = f.rfind('.')
    if f[dot_ind:] == '.txt':
      vid_num = f[4:7]
      lmdb_file_x = os.path.join(data_out_dir,vid_num+'X')
      lmdb_env_x = lmdb.open(lmdb_file_x, map_size=int(1e12))
      lmdb_txn_x = lmdb_env_x.begin(write=True)
      lmdb_file_y = os.path.join(data_out_dir,vid_num+'y')
      lmdb_env_y = lmdb.open(lmdb_file_y, map_size=int(1e12))
      lmdb_txn_y = lmdb_env_y.begin(write=True)
      video_path = os.path.join(video_dir,'vid_'+vid_num+'.avi')
      sample_num = 0

      cap = cv2.VideoCapture(video_path)
      total_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

      y_list = np.array([0]*total_num)
      #set up label
      y_file = open(os.path.join(annotations_dir,f),'r')
      for line in y_file:
        sp = line.split()
        y_list[int(sp[0]):int(sp[1])] = int(sp[2])

      for frame_num in range(0,total_num,skip_rate):
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame_num)
        ret,img = cap.read()
        if not ret:
          break
        img = cv2.resize(img,(width,height))
        
        keystr = '{:0>8d}'.format(sample_num)
        sample_num += 1
        img = img.astype(np.uint8)
        lmdb_txn_x.put( keystr.encode(), img.tobytes() )
      
        if (sample_num % batch) == 0:
          lmdb_txn_x.commit()
          lmdb_txn_x = lmdb_env_x.begin(write=True)
          print("name:{:s}  ,wrote: {:.3f} \r".format(f, frame_num*1.0/total_num))
          sys.stdout.flush()
      if (sample_num % batch) != 0:
        lmdb_txn_x.commit()
        print("name:{:s}  ,wrote: {:.3f} \n".format(f, frame_num*1.0/total_num))

      ## WRITE Y
      keystr = '{:0>8d}'.format(0)
      lmdb_txn_y.put( keystr.encode(), y_list.tobytes() )
      lmdb_txn_y.commit()
      
      cap.release()


def main():
  video_dir = "/home/tk/dev/data/CataractMount/Pilot2015/processed/videos/procedures/"
  annotations_dir = "/home/tk/dev/data/CataractMount/Pilot2015/processed/annotations/tasks/lfang6/"
  data_out_dir = '/media/tk/EE44DA8044DA4B4B/cataract_phase'
  weight_path = './weights/rescat_split0_aux/042_0.977.hdf5'

  files = os.listdir(annotations_dir)
  
  height = 135
  width = 240
  max_length = 184373/2

  skip_rate = 1 
  lmdb_out_batch = 128
  batch = 128
  curr_ind = 0

  n_classes = 21
  input_shape = (height,width,3)
  feat_dim = 128
  model = Models.ResCat_aux(n_classes,input_shape,dropout=0.5,aux_ind=1)
  model.load_weights(weight_path)

  for f in files:
    dot_ind = f.rfind('.')
    if f[dot_ind:] == '.txt':
      vid_num = f[4:7]
      lmdb_file_x = os.path.join(data_out_dir,vid_num)
      lmdb_env_x = lmdb.open(lmdb_file_x, map_size=int(1e12))
      lmdb_txn_x = lmdb_env_x.begin(write=True)
      
      video_path = os.path.join(video_dir,'vid_'+vid_num+'.avi')

      sample_num = 0
      cap = cv2.VideoCapture(video_path)
      total_num = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
      video_X = np.zeros((total_num,feat_dim))

      X = np.zeros((batch,height,width,3))
      for frame_num in range(0,total_num,skip_rate):
        cap.set(cv2.CAP_PROP_POS_FRAMES,frame_num)
        ret,img = cap.read()
        if not ret:
          break
        img = cv2.resize(img,(width,height))
        

        X[frame_num%batch] = (img.astype(np.float) - 128)/128.0

        if (frame_num+1) % batch == 0:
          feature = get_activations(model,-3, X)[0][0]
          video_X[frame_num-batch+1:frame_num] = feature
          X = np.zeros((batch,height,width,3))
          #pdb.set_trace()
        
        print("\r{:06d} / {:06d}".format(frame_num,total_num),end='\r')
        #pdb.set_trace()
      cap.release()  
      keystr = '{:0>8d}'.format(0)
      lmdb_txn_x.put( keystr.encode(), video_X.tobytes() )
      lmdb_txn_x.commit()
      lmdb_txn_x = lmdb_env_x.begin(write=True)
      #  sample_num += 1
      #  if sample_num % lmdb_out_batch == 0:
      #    lmdb_txn_x.commit()
          #lmdb_txn_x = lmdb_env_x.begin(write=True)
          #print("\r{:06d} / {:06d}".format(frame_num,total_num),end='\r')
      #if sample_num % lmdb_out_batch != 0:
        #lmdb_txn_x.commit() # last batch
      

      print("\nProcessed: ", f)

def get_activations(model, layer, X_batch):
  get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output,])
  activations = get_activations([X_batch,0])
  return activations

def test():
  data_out_dir = '/media/tk/EE44DA8044DA4B4B/cataract_phase'
  lmdb_file_train_x = os.path.join(data_out_dir,'034')
  lmdb_env_x = lmdb.open(lmdb_file_train_x)
  lmdb_txn_x = lmdb_env_x.begin()
  lmdb_cursor_x = lmdb_txn_x.cursor()

  for index in range(0,500,2):
    value = np.frombuffer(lmdb_cursor_x.get('{:0>8d}'.format(index).encode()),dtype=np.dtype(np.float))  
    pdb.set_trace()
    X = value.reshape((int(value.shape[0]/128.0),128))
    plt.imshow(X)
    plt.axis('tight')
    #image = value.reshape(135,240,3)
    cv2.imshow('test',image)
    if cv2.waitKey(1) == ord('q'):
      cv2.destroyAll()     

if __name__ == "__main__":
  #main()
  #test()
  #test_imgnet()
  #write_phase_image()
  write_phase_tcn()
