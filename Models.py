import numpy as np
import pdb
import six
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

import resnet2


#Assuming TF
ROW_AXIS=1
COL_AXIS=2
CHANNEL_AXIS = 3


def resnet(nb_classes):
  model = resnet2.ResNet50(weights='imagenet', input_shape=(224, 224, 3),
             pooling='avg',
             classes=nb_classes)
  return model


def ResCat_aux(n_classes,input_shape,dropout=0.5,aux_ind=0):
  filter_numbers = [32,64,128]
  init_filters = 32
  merge_layers = []
  outputs = []
  input = Input(shape=input_shape)
  conv1 = Conv2D(filters=init_filters, kernel_size=(7,7),
                 strides=(2,2), padding='valid',
                 kernel_initializer='he_normal',
                 kernel_regularizer=l2(1.e-4))(input)
  norm = BatchNormalization(axis=CHANNEL_AXIS)(conv1)
  relu = Activation("relu")(norm)
  dr = Dropout(dropout)(relu)
  model = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same")(dr)

  for fn in filter_numbers:
    res1 = bn_relu_conv(model,fn,(3,3),dropout=dropout)
    res2 = bn_relu_conv(res1,fn,(3,3),strides=2,dropout=dropout)

    input_shape = K.int_shape(model)
    residual_shape = K.int_shape(res2)
    stride_width = int(round(input_shape[ROW_AXIS] / residual_shape[ROW_AXIS]))
    stride_height = int(round(input_shape[COL_AXIS] / residual_shape[COL_AXIS]))
    equal_channels = input_shape[CHANNEL_AXIS] == residual_shape[CHANNEL_AXIS]

    # 1 X 1 conv if shape is different. Else identity.
    if stride_width > 1 or stride_height > 1 or not equal_channels:
      model = Conv2D(filters=residual_shape[CHANNEL_AXIS],
                        kernel_size=(1, 1), 
                        strides=(stride_width, stride_height),
                        padding="valid",
                        kernel_initializer="he_normal",
                        kernel_regularizer=l2(0.0001))(model)
    model = add([res2,model])
    merge_layers.append(model)
    # Last activation
  norm = BatchNormalization(axis=CHANNEL_AXIS)(model)
  model = Activation("relu")(norm)
  # Classifier block
  block_shape = K.int_shape(model)
  model = AveragePooling2D(pool_size=(block_shape[ROW_AXIS], block_shape[COL_AXIS]),
                           strides=(1, 1))(model)
  flatten = Flatten()(model)
  dense = Dense(units=n_classes,
                kernel_initializer="he_normal",
                activation="sigmoid",
                name='main')(flatten)

  outputs.append(dense)
  ## AUX, does obj exists?
  aux = merge_layers[aux_ind]
  pool = AveragePooling2D(pool_size=(3,3),
                           strides=(2, 2))(aux)
  flatten = Flatten()(pool)
  fc = Dense(units=filter_numbers[-1],
                kernel_initializer="he_normal",
                activation="relu")(flatten)
  dense = Dense(units=2,
                kernel_initializer="he_normal",
                activation="softmax",
                name='aux')(fc)
  outputs.append(dense)

  model = Model(inputs=input, outputs=outputs)
  model.summary()
  return model


def bn_relu_conv(blob,num_filters,
                 conv_shape,
                 strides=1,
                 padding='same',
                 kernel_initializer='he_normal',
                 kernel_regularizer=l2(1.e-4),
                 dropout=0.5):

  norm = BatchNormalization(axis=CHANNEL_AXIS)(blob)
  relu = Activation("relu")(norm)
  dr = Dropout(dropout)(relu)
  return Conv2D(filters=num_filters, kernel_size=conv_shape,
                strides=strides, padding=padding,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=kernel_regularizer)(dr)