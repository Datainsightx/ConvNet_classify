# ConvNet_classify
A deep learning classification of images using ConvNet
#CNN deep learning algorithm compiled by Isaac Alabi
#Adaptation is from the keras documentation
#================================================================================
#Import the modules/libraries needed for the ConvNet algorithm

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

#=================================================================================
#Build CNN model
#=================================================================================
#Create a sequential model by passing a list of layer instances

model = Sequential()
model.add(Conv2D(32, 3, 3, input_shape=(3, 150, 150)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

#===============================================================================
# Configure the learning process

model.compile(loss='binary_crossentropy',
              optimizer='RMSprop',
              metrics=['accuracy'])

#===================================================================================
# Generate batches of tensor image data with real-time data augmentation.
#The data will be looped over (in batches) indefinitely.

batch_size = 32

datagen = ImageDataGenerator(
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False,
        fill_mode='nearest',
        featurewise_center=True,
        featurewise_std_normalization=True,
        zca_whitening=False)

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
        rescale=1./255.0,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=False)

test_datagen = ImageDataGenerator(rescale=1./255.0)

train_generator = train_datagen.flow_from_directory(
        "/home/isaacalabi/Data_cp/train/",
        target_size=(150, 150),
        batch_size=batch_size,
        shuffle=True,
        class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
        "/home/isaacalabi/Data_cp/test/",
        target_size=(150, 150),
        batch_size=batch_size,
        shuffle=True,
        class_mode='binary')


#================================================================================================
#Train model
#================================================================================================
nb_train_samples = 708
nb_validation_samples = 112
epochs = 50
batch_size = 32

checkpointer = ModelCheckpoint(filepath="/tmp/weights.hdf5", verbose=1, save_best_only=True)

model.fit_generator(generator=train_generator,
        validation_data=validation_generator,
        samples_per_epoch=nb_train_samples,
        nb_epoch=epochs,
        nb_val_samples=nb_validation_samples,
        callbacks=[checkpointer])

#================================================================================================
#Save model weights for future prediction tasks

model.save_weights('cookie.h5')
