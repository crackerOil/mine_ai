import tensorflow as tf
from keras.utils import np_utils
import numpy as np

from PIL import Image

import image_values

def load_image(filename):
	img = Image.open(filename)
	img.load()
	img = img.resize((40, 40))
	data = np.asarray(img, dtype="int32")

	return data
	
def load_training_set(directory):
	dataset = []
	
	for i in range(1, 43):
		dataset.append(load_image(directory + "\\" + str(i) + ".png"))
		
	return np.array(dataset)

def model():
	model = tf.keras.Sequential()
	
	model.add(tf.keras.layers.Conv2D(filters = 6, kernel_size = 5,
									 strides = 1, padding = "same",
									 data_format = "channels_last",
									 activation = tf.nn.relu,
									 input_shape = [40, 40, 3]))
									 
	model.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2, padding = "valid",
										data_format = "channels_last",
										input_shape = [40, 40, 6]))
										
	model.add(tf.keras.layers.Conv2D(filters = 16, kernel_size = 5,
									 strides = 1, padding = "valid",
									 data_format = "channels_last",
									 activation = tf.nn.relu,
									 input_shape = [20, 20, 6]))
									 
	model.add(tf.keras.layers.MaxPool2D(pool_size = 2, strides = 2, padding = "valid",
										data_format = "channels_last",
										input_shape = [16, 16, 16]))
										
	model.add(tf.keras.layers.Flatten())
	#model.add(tf.keras.layers.Dense(128, activation = tf.nn.relu, use_bias = True))
	model.add(tf.keras.layers.Dense(64, activation = tf.nn.relu, use_bias = True))
	model.add(tf.keras.layers.Dense(13, activation = tf.nn.softmax, use_bias = True))
	
	return model
	
def train_model(training_dir):
	x_train = load_training_set(training_dir)
	x_train = x_train / 255
	
	y_train = np.array(image_values.y_values)
	y_train = np_utils.to_categorical(y_train)
	
	ai_model = model()
	ai_model.compile(optimizer = tf.train.RMSPropOptimizer(0.001, 0.9),
					 loss = "categorical_crossentropy", metrics = ["accuracy"])
	
	ai_model.summary()
	
	ai_model.fit(x_train, y_train, epochs=100)
	
	ai_model.save("minesweeper_tile_recognition_2.model")
	
train_model("training_imgs")