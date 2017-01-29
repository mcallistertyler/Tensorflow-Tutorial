#Tensorflow Convolutional Neural Network
#A simple Convolutional Neural Network
%matplotlib inline
import matplotlib.pyplot as plt
import tensorflow as tf 
import numpy as np 
from sklearn.metrics import confusion_matrix
import time
from datetime import timedelta
import math

#Configuration of Neural Network
#Layer 1
filter_size1 = 5		#Convolution filters are 5 x 5 pixels.
num_filters1 = 16		#There are 16 of these filters

#Layer 2
filter_size2 = 5		#Convolution filters are 5 x 5 pixels.
num_filters2 = 36		#There are 36 of these filters

#Fully-connected layer
fc_size = 128			#Number of neurons in fully-connected layer.

from tensorflow.examples.tutorials.mnist import input_data
data = input_data.read_data_sets('data/MNIST/', one_hot=True)

print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(data.test.labels)))
print("- Validation-set:\t{}".format(len(data.validation.labels)))

data.test.cls = np.argmax(data.test.labels, axis=1)

#MNIST images are 28 pixels in dimension.
img_size = 28

#Images are stored in one-dimensional arrays of this length
img_size_flat = img_size * img_size

#Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

#Number of colour channels for the images: 1 channel for gray-scale
num_channels = 1

#Number of classes, once class for each of 10 digits/
num_classes = 10

def plot_images(images, cls_true, cls_pred=None):
	assert len(images) == len(cls_true) == 9

	#Create figure with 3x3 sub-plots.
	fig, axes = plt.subsplots(3,3)
	fig.subsplots_adjust(hspace=0.3, wspace=0.3)

	for i, ax in enumerate(axes.flat):
		#Plot image.
		ax.imshow(images[i].reshape(img_shape), cmap='binary')

		#Show true and predicted classes
		if cls_pred is None:
			xlabel = "True: {0}".format(cls_true[i])
		else:
			xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

		#Show the classes as the label on the x-axis
		ax.set_xlabel(xlabel)

		#Remove the ticks from the plot
		ax.set_xticks([])
		ax.set_yticks([])

	#Ensure the plot is shown correctly with multiple plots
	#in a single Notebook cell.
	plt.show()

#Get the first images from the test-set
images = data.test.images[0:9]

#Get the true classes for those images.
cls_true = data.test.cls[0:9]

#Plot the images and labels using our helper-function above.
plot_images(images=images, cls_true=cls_true)

def new_weights(shape):
	return tf.Variable(tf.truncated_normal(shape, stddev=0.05))

def new_biases(length):
	return tf.Variable(tf.constant(0.05, shape=[length]))

def new_conv_layer(input, num_input_channels, filter_size, num_filters,	use_pooling=True):

	#Shape of the filter-weights for the convolution
	#This format is determined by the Tensorflow API
	shape = [filter_size, filter_size, num_input_channels, num_filters]

	#Create the new weights aka. filters with the given shape
	weights = new_weights(shape=shape)
	#Create the new biases, one for each filter.
	biases = new_biases(length=num_filters)

	#Create the Tensorflow operation for convolution
	#Note the strides are set to 1 in all dimensions.
	#The first and last stride must always be 1,
	#because the first is for the image-number and
	#the last is for the input-channel.
	#But e.g strides=[1,2,2,1] would mean that the filter
	#is moved 2 pixels across the x- and y-axis of the image
	#The padding is set to SAME which means the input image
	#is padded with zeroes so the size of the output is the same.
	layer = tf.nn.conv2d(input=input, filter=weights, stride=[1,1,1,1], padding='SAME')

	#Add the biases to the results of the convolution.
	#A bias-value is added to each filter-channel.
	layer += biases

	#Use pooling to down-sample the image resolution?
	if use_pooling:
		#This is 2x2 max-pooling, which means that we
		#consider 2x2 windows and select the largest value
		#in each window. Then we move 2 pixels to the next window.
		layer = tf.nn.max_pool(value=layer, ksze=[1,2,2,1], strides=[1,2,2,1], padding = 'SAME')

	layer tf.nn.relu(layer)

	return layer, weights

def flatten_layer(layer):
	#Get the shape of the input layer.
	layer_shape = layer.get_shape()

	#The shape of the input layer is assumed to be:
	#layer_shape == [num_images, img_width, img_height, num_channels]

	#The number of features is: img_width * img_height * num_channels
	#The shape uses a TensorFlow datatype, so convert it to numpy.
	num_features = np.array(layer_shape[1:4], dtype=int).prod()

	# Reshape the layer to [num_images, num_features].
    # Note that we just set the size of the second dimension
    # to num_features and the size of the first dimension to -1
    # which means the size in that dimension is calculated
    # so the total size of the tensor is unchanged from the reshaping.
    layer_flat = tf.reshape(layer, [-1, num_features])

    # The shape of the flattened layer is now:
    # [num_images, img_height * img_width * num_channels]

    # Return both the flattened layer and the number of features.
    return layer_flat, num_features