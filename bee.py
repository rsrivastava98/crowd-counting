import numpy as np
from keras.applications.vgg16 import VGG16
import tensorflow as tf
import os
import preprocessing

train_images = np.array(preprocessing.image_patches("data/shanghaitech_h5_empty/ShanghaiTech/part_A/train_data/images"))
test_images = np.array(preprocessing.image_patches("data/shanghaitech_h5_empty/ShanghaiTech/part_A/test_data/images"))
print(type(test_images))
print(test_images.shape)
print(test_images[0])

model = VGG16()
model.load_weights('/vgg16_imagenet.h5')



train_labels = np.ones(2700)
test_labels = np.ones(1638)

checkpoint_path = "training_1/cp.h5"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)

# Train the model with the new callback
model.fit(train_images, 
          train_labels,  
          epochs=10,
          validation_data=(test_images,test_labels),
          callbacks=[cp_callback])  # Pass callback to training

# This may generate warnings related to saving the state of the optimizer.
# These warnings (and similar warnings throughout this notebook)
# are in place to discourage outdated usage, and can be ignored.

print(model.summary())
