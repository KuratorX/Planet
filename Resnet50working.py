#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 17:23:11 2017

@author: chris
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
import gc

import keras as k
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau, CSVLogger, EarlyStopping
import cv2
from tqdm import tqdm
from keras.regularizers import l2
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import KFold
from sklearn.metrics import fbeta_score
import time
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam , RMSprop, Adadelta, SGD
x_train = []
x_test = []
y_train = []

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('sample_submission.csv')

flatten = lambda l: [item for sublist in l for item in sublist]
labels = list(set(flatten([l.split(' ') for l in df_train['tags'].values])))

labels = ['blow_down',
 'bare_ground',
 'conventional_mine',
 'blooming',
 'cultivation',
 'artisinal_mine',
 'haze',
 'primary',
 'slash_burn',
 'habitation',
 'clear',
 'road',
 'selective_logging',
 'partly_cloudy',
 'agriculture',
 'water',
 'cloudy']

label_map = {'agriculture': 14,
 'artisinal_mine': 5,
 'bare_ground': 1,
 'blooming': 3,
 'blow_down': 0,
 'clear': 10,
 'cloudy': 16,
 'conventional_mine': 2,
 'cultivation': 4,
 'habitation': 9,
 'haze': 6,
 'partly_cloudy': 13,
 'primary': 7,
 'road': 11,
 'selective_logging': 12,
 'slash_burn': 8,
 'water': 15}

'''
for f, tags in tqdm(df_train.values, miniters=1000):
    img = cv2.imread('train-jpg/{}.jpg'.format(f))
    
    targets = np.zeros(17)
    for t in tags.split(' '):
        targets[label_map[t]] = 1 
    x_train.append(cv2.resize(img, (224,224)))
   # y_train.append(targets)

#y_train = np.array(y_train, np.uint8)

#x_train = np.array(x_train)
x_train = np.array(x_train, np.float32) / 255.


#x_train = np.array(x_train, np.float32) / 255.


np.save('x_train.npy', x_train)

'''


y_train = np.load('y_train.npy')



print(y_train.shape)


import numpy as np
from sklearn.metrics import fbeta_score

def optimise_f2_thresholds(y, p, verbose=True, resolution=100):
  def mf(x):
    p2 = np.zeros_like(p)
    for i in range(17):
      p2[:, i] = (p[:, i] > x[i]).astype(np.int)
    score = fbeta_score(y, p2, beta=2, average='samples')
    return score

  x = [0.2]*17
  for i in range(17):
    best_i2 = 0
    best_score = 0
    for i2 in range(resolution):
      i2 /= float(resolution)
      x[i] = i2
      score = mf(x)
      if score > best_score:
        best_i2 = i2
        best_score = score
    x[i] = best_i2
    if verbose:
      print(i, best_i2, best_score)

  return x


from keras.layers.normalization import BatchNormalization

nfolds = 5

num_fold = 0
sum_score = 0

yfull_test = []
yfull_train =[]
np.random.seed(1337)
allthresh = []
w_regularizer = l2(10**-7)
from sklearn.model_selection import StratifiedShuffleSplit
kf = KFold(len(y_train), n_folds=nfolds, shuffle=True, random_state=1)
sss = StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
for train_index, test_index in kf:#sss.split(x_train, y_train):#kf:
        gc.collect()
        start_time_model_fitting = time.time()
        num_fold += 1
       # if num_fold == 1 or num_fold == 2 or num_fold == 3:
        #    continue
        #print(num_fold)
        
        #if num_fold == 4:
         #   yfull_test = list(np.load('cvr3.npy'))
          #  allthresh =  list(np.load('thr3.npy'))
        x_train = np.load('x_train.npy')

        X_train = x_train[train_index]
        Y_train = y_train[train_index]
        X_valid = x_train[test_index]
        Y_valid = y_train[test_index]
        
        
        del x_train
        gc.collect()
       # num_fold += 1
        print('Start KFold number {} from {}'.format(num_fold, nfolds))
        print('Split train: ', len(X_train), len(Y_train))
        print('Split valid: ', len(X_valid), len(Y_valid))

      
       # model.compile(loss='binary_crossentropy', 
                     # optimizer='Adam',
#                      metrics=['accuracy'])
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=7, verbose=0),
            ReduceLROnPlateau( monitor='val_loss',factor=0.1, cooldown=0, patience=3, min_lr=0.5e-8),
            ModelCheckpoint('weights.h5', monitor='val_loss', save_best_only=True, verbose=0)]
        
        
        from keras.preprocessing.image import ImageDataGenerator

        datagen = ImageDataGenerator(
                      
                       # featurewise_center=True,
                        #featurewise_std_normalization=True,
                      # samplewise_center = True,
                        #samplewise_std_normalization=True,
                           #rotation_range=90,
                           zoom_range=0.1,
                       # rescale=1./255,
                           vertical_flip = True,
                           horizontal_flip = True,
                           width_shift_range=0.1,
                           height_shift_range=0.1,
                           shear_range=0.2,
                       # zca_whitening=True, 
                       fill_mode="reflect")
        
        
        from keras.applications.inception_v3 import InceptionV3
        from keras.applications.vgg19 import VGG19
        from keras.applications.vgg16 import VGG16
        from keras.applications.xception import Xception
        from keras.preprocessing import image
        from keras.applications.vgg16 import preprocess_input
        from keras.layers import Input, Flatten, Dense
        from keras.models import Model
        import numpy as np
        
        from keras.applications.resnet50 import ResNet50
        from keras.preprocessing import image
        from keras.applications.resnet50 import preprocess_input, decode_predictions
        from keras.layers import Dense, GlobalAveragePooling2D
        from keras import backend as K
        
        
        base_model =  ResNet50(weights='imagenet', include_top=False, input_shape=(224,224,3))
        
        # add a global spatial average pooling layer
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        # let's add a fully-connected layer
        #x = Flatten()(x)
        x = Dense(1024,W_regularizer=w_regularizer)(x)
        x = PReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(1024,W_regularizer=w_regularizer)(x)
        x = PReLU()(x)
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)

        # and a logistic layer -- let's say we have 200 classes
        predictions = Dense(17, activation='sigmoid',W_regularizer=w_regularizer)(x)
        
        # this is the model we will train
        model = Model(inputs=base_model.input, outputs=predictions)
        
        for layer in base_model.layers:
            layer.trainable = False
        
        # compile the model (should be done *after* setting layers to non-trainable)
        model.compile(optimizer='adam', loss='binary_crossentropy')
        
        # train the model on the new data for a few epochs
       # model.fit(X_train, y= Y_train,
        #                batch_size=128,verbose=1, nb_epoch=7,validation_split=0.1, shuffle=True)

        model.fit_generator(datagen.flow(X_train,Y_train,shuffle=True, batch_size=128), validation_data=(X_valid, Y_valid),
                 samples_per_epoch=len(X_train), nb_epoch=100, verbose=1,callbacks=callbacks )
        
        
        #model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy')
        
        
       # model.fit_generator(datagen.flow(X_train,Y_train,shuffle=True, batch_size=128), validation_data=(X_valid, Y_valid),
               #  samples_per_epoch=len(X_train), nb_epoch=100, verbose=1,callbacks=callbacks )
        
        model.load_weights('weights.h5')
        model.save_weights("modelweights.h5")
        model.load_weights("modelweights.h5")
        
        
        
        
      #  for i, layer in enumerate(base_model.layers):
          # print(i, layer.name)
        
        # we chose to train the top 2 inception blocks, i.e. we will freeze
        # the first 172 layers and unfreeze the rest:
        #for layer in model.layers[:140]:
           #layer.trainable = False
        for layer in model.layers:
           layer.trainable = True
        
        # we need to recompile the model for these modifications to take effect
        # we use SGD with a low learning rate
        from keras.optimizers import SGD #SGD(lr=0.001, momentum=0.9)
        model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy')
        
        # we train our model again (this time fine-tuning the top 2 inception blocks
        # alongside the top Dense layers
       # model.fit(x_train, y= Y_train,
                       # batch_size=64,verbose=1, nb_epoch=40,validation_split=0.1, shuffle=True)
        model.fit_generator(datagen.flow(X_train,Y_train,shuffle=True, batch_size=64), validation_data=(X_valid, Y_valid),
                 samples_per_epoch=len(X_train), nb_epoch=9, verbose=1 ,callbacks=callbacks)
        
        
        model.load_weights('weights.h5')
        model.compile(optimizer=Adam(lr=1e-5), loss='binary_crossentropy')
        
        # we train our model again (this time fine-tuning the top 2 inception blocks
        # alongside the top Dense layers
       # model.fit(x_train, y= Y_train,
                       # batch_size=64,verbose=1, nb_epoch=40,validation_split=0.1, shuffle=True)
       # model.load_weights('weights.h5')
        model.fit_generator(datagen.flow(X_train,Y_train,shuffle=True, batch_size=64), validation_data=(X_valid, Y_valid),
                 samples_per_epoch=len(X_train), nb_epoch=4, verbose=1 ,callbacks=callbacks)
        
        print('Loading saved weights...')
        print('-'*30)
        model.load_weights('weights.h5')
        
        

        p_valid = model.predict(X_valid, batch_size = 128, verbose=2)
        
        
        
        print(fbeta_score(Y_valid, np.array(p_valid) > 0.2, beta=2, average='samples'))
        
        x = optimise_f2_thresholds(Y_valid, p_valid)
        
       # p_valid = model.predict_generator(datagen.flow(X_valid,shuffle=False, batch_size = 128) , 64)
        
        
        
        #print(fbeta_score(Y_valid, np.array(p_valid) > 0.2, beta=2, average='samples'))
        
        
        nbr_augmentation = 15
        for idx in range(nbr_augmentation):
            print('{}th augmentation for testing ...'.format(idx))
            random_seed = np.random.random_integers(0, 100000)
           
              
            datagen = ImageDataGenerator(
                      
                       # featurewise_center=True,
                        #featurewise_std_normalization=True,
                      # samplewise_center = True,
                        #samplewise_std_normalization=True,
                          # rotation_range=90,
                           zoom_range=0.1,
                       # rescale=1./255,
                           vertical_flip = True,
                           horizontal_flip = True,
                           width_shift_range=0.1,
                           height_shift_range=0.1,
                           shear_range=0.2,
                        #zca_whitening=True, 
                       fill_mode="reflect")
            #p_valid = model.predict(X_valid, batch_size = 128, verbose=2)
        
        
            #test_image_list = test_generator.filenames
            #print('image_list: {}'.format(test_image_list[:10]))
            print('Begin to predict for testing data ...')
            if idx == 0:
                p_test = model.predict_generator(datagen.flow(X_valid,shuffle = False, batch_size = 128) , 64)
                
                
                #print('Begin to predict for testing data ...')

            else:
               p_test += model.predict_generator(datagen.flow(X_valid, shuffle = False , batch_size = 128) , 64)
               
               
               
          
    
        p_test /= nbr_augmentation
        
        
    
        x = optimise_f2_thresholds(Y_valid, p_test)
       
        allthresh.append(x)
        gc.collect()
        del X_train
        gc.collect()
    
        del X_valid
        gc.collect()
        #print(fbeta_score(Y_valid, np.array(p_valid) > x, beta=2, average='samples'))
        print("Optimizing prediction threshold")
        
        
        nbr_augmentation = 15
        for idx in range(nbr_augmentation):
            print('{}th augmentation for testing ...'.format(idx))
            random_seed = np.random.random_integers(0, 100000)
           
              
            datagen = ImageDataGenerator(
                      
                       # featurewise_center=True,
                        #featurewise_std_normalization=True,
                      # samplewise_center = True,
                        #samplewise_std_normalization=True,
                          # rotation_range=90,
                           zoom_range=0.1,
            
                           vertical_flip = True,
                           horizontal_flip = True,
                           width_shift_range=0.1,
                           height_shift_range=0.1,
                           shear_range=0.2,
                        #zca_whitening=True, 
                       fill_mode="reflect")
           
            
            
            print('Begin to predict for testing data ...')
            if idx == 0:
                print("01")
              
                x_test1  = np.load('x1.npy')
                p_test1 = model.predict_generator(datagen.flow(x_test1, shuffle = False, batch_size = 128) , 157)
                gc.collect()
                del x_test1
                gc.collect()
                print("2")
                x_test2  = np.load('x2.npy')
                p_test2 = model.predict_generator(datagen.flow(x_test2, shuffle = False, batch_size = 128) , 157)
                gc.collect()
                del x_test2
                gc.collect()
                print("3")
                x_test3  = np.load('x3.npy')
                print(x_test3.shape)
                p_test3 = model.predict_generator(datagen.flow(x_test3, shuffle = False, batch_size = 128) , 166)
                gc.collect()
                del x_test3
                gc.collect()
                p_test = np.vstack((p_test1,p_test2,p_test3))
                print('Begin to predict for testing data ...')

            else:
                print("1")
                x_test1  = np.load('x1.npy')
                p_test1 = model.predict_generator(datagen.flow(x_test1, shuffle = False, batch_size = 128) , 157)
                gc.collect()
                del x_test1
                gc.collect()
                print("2")
                x_test2  = np.load('x2.npy')
                p_test2 = model.predict_generator(datagen.flow(x_test2, shuffle = False, batch_size = 128) , 157)
                gc.collect()
                del x_test2
                gc.collect()
                print("3")
                x_test3  = np.load('x3.npy')
                p_test3 = model.predict_generator(datagen.flow(x_test3, shuffle = False, batch_size = 128) , 166)
                gc.collect()
                del x_test3
                gc.collect()
                p_test += np.vstack((p_test1,p_test2,p_test3))
               
               
    
        p_test /= nbr_augmentation
        
        
      
        
        yfull_test.append(p_test)
        gc.collect()
        del p_test
        gc.collect()
        
        
#yfull_test = list(np.load('cvr4.npy'))        
result = np.array(yfull_test[0])
for i in range(1, nfolds):
    result += np.array(yfull_test[i])
result /= nfolds
result = pd.DataFrame(result, columns = labels)
result


np.save('Resnetaug-224-0.npy', result)
from tqdm import tqdm

#thres = [0.2] * 17
#thres = (np.asarray(thres1)+np.asarray(thres2))/2
thres =  np.asarray(allthresh).mean(axis=0)
np.save('Resnetaug-224-0-th.npy', thres)
#thres = [0.2] * 17
preds = []
for i in tqdm(range(result.shape[0]), miniters=1000):
    a = result.ix[[i]]
    a = a.apply(lambda x: x > thres, axis=1)
    a = a.transpose()
    a = a.loc[a[i] == True]
    ' '.join(list(a.index))
    preds.append(' '.join(list(a.index)))
    
df_test['tags'] = preds
df_test

df_test.to_csv('submission_keras.csv', index=False)


