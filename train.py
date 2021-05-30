import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D, AveragePooling2D, Input, BatchNormalization
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Flatten, Dense, Dropout, Conv2D, MaxPooling2D, AveragePooling2D, Input, BatchNormalization
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import splitfolders
import os

if not os.path.isdir('./data'):
    path = '/Users/steph/cs_face_recognition/data'
    DATA_PATH = './data'
    splitfolders.ratio(path, DATA_PATH, seed=42, ratio=(.8, 0.1,0.1))
DATA_PATH = './data'
train_dir = os.path.join(DATA_PATH, 'train')
val_dir = os.path.join(DATA_PATH, 'val')

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224,224,3))
for layer in base_model.layers:
  layer.trainable = False

x = base_model.get_layer('block5_pool').output
x = Conv2D(64, 3)(x)
x = MaxPooling2D(pool_size=(2, 2))(x)
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(4, activation='softmax')(x)

vgg = keras.Model(inputs=base_model.input, outputs=x)
vgg.summary()

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
    preprocessing_function=preprocess_input
    )

val_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_generator = datagen.flow_from_directory(
                      train_dir,
                      target_size=(224, 224), # check required input shape
                      batch_size=32,
                      # shuffle=True,
                      class_mode='categorical')

val_generator = val_datagen.flow_from_directory(
                      val_dir,
                      target_size=(224, 224), # check required input shape
                      batch_size=32,
                      shuffle=False,          # IMPORTANT!
                      class_mode='categorical')


# Callback functions     
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath="my_model_checkpoint_{epoch}.h5",
                                                 save_weights_only=False, # the whole model (False) or only weights (True) 
                                                 save_best_only=True, # keep the best model with lowest validation loss
                                                 monitor='val_loss',
                                                 verbose=1)
earlystopping_callback = tf.keras.callbacks.EarlyStopping(
    # Stop training when `val_loss` is no longer improving
    monitor='val_loss',
    # "no longer improving" being defined as "no better than 1e-2 less"
    min_delta=1e-2,
    # "no longer improving" being further defined as "for at least 5 epochs", 
    # you can choose a lower number if you are impatient
    patience=5,
    verbose=1)

vgg.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])
history = vgg.fit(train_generator, 
                    validation_data=val_generator,
                    callbacks=[checkpoint_callback,earlystopping_callback],
                    epochs=10)