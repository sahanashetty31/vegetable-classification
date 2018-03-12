
# coding: utf-8

# In[456]:


import time
import math
import random

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import dataset
import cv2

from sklearn.metrics import confusion_matrix
from datetime import timedelta

get_ipython().magic('matplotlib inline')


# In[457]:


validation_size = .8


early_stopping = None 


# Convolutional Layer 1.
filter_size1 = 3 
num_filters1 = 32

# Convolutional Layer 2.
filter_size2 = 3
num_filters2 = 32

# Convolutional Layer 3.
filter_size3 = 3
num_filters3 = 64

# Fully-connected layer.
fc_size = 128             # Number of neurons in fully-connected layer.

# Number of color channels for the images: 1 channel for gray-scale.
num_channels = 3

# image dimensions (only squares for now)
#img_size = 128
img_size = 32

# Size of image when flattened to a single dimension
img_size_flat = img_size * img_size * num_channels

# Tuple with height and width of images used to reshape arrays.
img_shape = (img_size, img_size)

# class info

#For differentiating vegetables
classes = ['onion', 'tomato', 'others']



num_classes = len(classes)

# batch size
#batch_size = 32
batch_size = 8


train_path = 'data1/train/'
test_path = 'data1/test/'




checkpoint_dir = "models/"


# In[458]:


data = dataset.read_train_sets(train_path, img_size, classes, validation_size=validation_size)
test_images, test_ids = dataset.read_test_set(test_path, img_size)


# In[459]:


print("Size of:")
print("- Training-set:\t\t{}".format(len(data.train.labels)))
print("- Test-set:\t\t{}".format(len(test_images)))
print("- Validation-set:\t{}".format(len(data.valid.labels)))


# In[460]:


def plot_images(images, cls_true, cls_pred=None):
    
    if len(images) == 0:
        print("no images to show")
        return 
    else:
        random_indices = random.sample(range(len(images)), min(len(images), 9))
        
        
    images, cls_true  = zip(*[(images[i], cls_true[i]) for i in random_indices])
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i].reshape(img_size, img_size, num_channels))

        # Show true and predicted classes.
        if cls_pred is None:
            xlabel = "True: {0}".format(cls_true[i])
        else:
            xlabel = "True: {0}, Pred: {1}".format(cls_true[i], cls_pred[i])

        # Show the classes as the label on the x-axis.
        ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()


# In[461]:


# Get some random images and their labels from the train set.

images, cls_true  = data.train.images, data.train.cls

# Plot the images and labels using our helper-function above.
plot_images(images=images, cls_true=cls_true)


# In[462]:


def new_weights(shape):
    return tf.Variable(tf.truncated_normal(shape, stddev=0.05))


# In[463]:


def new_biases(length):
    return tf.Variable(tf.constant(0.05, shape=[length]))


# In[464]:


def new_conv_layer(input,              # The previous layer.
                   num_input_channels, # Num. channels in prev. layer.
                   filter_size,        # Width and height of each filter.
                   num_filters,        # Number of filters.
                   use_pooling=True):  # Use 2x2 max-pooling.

    # Shape of the filter-weights for the convolution.
    # This format is determined by the TensorFlow API.
    shape = [filter_size, filter_size, num_input_channels, num_filters]

    # Create new weights aka. filters with the given shape.
    weights = new_weights(shape=shape)

    # Create new biases, one for each filter.
    biases = new_biases(length=num_filters)

    # Create the TensorFlow operation for convolution.
    # Note the strides are set to 1 in all dimensions.
    # The first and last stride must always be 1,
    # because the first is for the image-number and
    # the last is for the input-channel.
    # But e.g. strides=[1, 2, 2, 1] would mean that the filter
    # is moved 2 pixels across the x- and y-axis of the image.
    # The padding is set to 'SAME' which means the input image
    # is padded with zeroes so the size of the output is the same.
    layer = tf.nn.conv2d(input=input,
                         filter=weights,
                         strides=[1, 1, 1, 1],
                         padding='SAME')

    # Add the biases to the results of the convolution.
    # A bias-value is added to each filter-channel.
    layer += biases

    # Use pooling to down-sample the image resolution?
    if use_pooling:
        # This is 2x2 max-pooling, which means that we
        # consider 2x2 windows and select the largest value
        # in each window. Then we move 2 pixels to the next window.
        layer = tf.nn.max_pool(value=layer,
                               ksize=[1, 2, 2, 1],
                               strides=[1, 2, 2, 1],
                               padding='SAME')

    # Rectified Linear Unit (ReLU).
    # It calculates max(x, 0) for each input pixel x.
    # This adds some non-linearity to the formula and allows us
    # to learn more complicated functions.
    layer = tf.nn.relu(layer)

    # Note that ReLU is normally executed before the pooling,
    # but since relu(max_pool(x)) == max_pool(relu(x)) we can
    # save 75% of the relu-operations by max-pooling first.

    # We return both the resulting layer and the filter-weights
    # because we will plot the weights later.
    return layer, weights


# In[465]:


def flatten_layer(layer):
    # Get the shape of the input layer.
    layer_shape = layer.get_shape()

    # The shape of the input layer is assumed to be:
    # layer_shape == [num_images, img_height, img_width, num_channels]

    # The number of features is: img_height * img_width * num_channels
    # We can use a function from TensorFlow to calculate this.
    num_features = layer_shape[1:4].num_elements()
    
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


# In[466]:


def new_fc_layer(input,          # The previous layer.
                 num_inputs,     # Num. inputs from prev. layer.
                 num_outputs,    # Num. outputs.
                 use_relu=True): # Use Rectified Linear Unit (ReLU)?

    # Create new weights and biases.
    weights = new_weights(shape=[num_inputs, num_outputs])
    biases = new_biases(length=num_outputs)

    # Calculate the layer as the matrix multiplication of
    # the input and weights, and then add the bias-values.
    layer = tf.matmul(input, weights) + biases

    # Use ReLU?
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer


# In[467]:


x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')


# In[468]:


x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])


# In[469]:


y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')


# In[470]:


y_true_cls = tf.argmax(y_true, dimension=1)


# In[471]:


layer_conv1, weights_conv1 =     new_conv_layer(input=x_image,
                   num_input_channels=num_channels,
                   filter_size=filter_size1,
                   num_filters=num_filters1,
                   use_pooling=True)


# In[472]:


layer_conv1


# In[473]:


layer_conv2, weights_conv2 =     new_conv_layer(input=layer_conv1,
                   num_input_channels=num_filters1,
                   filter_size=filter_size2,
                   num_filters=num_filters2,
                   use_pooling=True)


# In[474]:


layer_conv2


# In[475]:


layer_conv3, weights_conv3 =     new_conv_layer(input=layer_conv2,
                   num_input_channels=num_filters2,
                   filter_size=filter_size3,
                   num_filters=num_filters3,
                   use_pooling=True)


# In[476]:


layer_conv3


# In[477]:


layer_flat, num_features = flatten_layer(layer_conv3)


# In[478]:


layer_flat


# In[479]:


num_features


# In[480]:


layer_fc1 = new_fc_layer(input=layer_flat,
                        num_inputs=num_features,
                        num_outputs=fc_size,
                        use_relu=True)


# In[481]:


layer_fc1


# In[482]:


layer_fc2 = new_fc_layer(input=layer_fc1,
                        num_inputs=fc_size,
                        num_outputs=num_classes,
                        use_relu=False)


# In[483]:


layer_fc2


# In[484]:


y_pred = tf.nn.softmax(layer_fc2)


# In[485]:


y_pred_cls = tf.argmax(y_pred, dimension=1)


# In[486]:


cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,
                                                        labels=y_true)


# In[487]:


cost = tf.reduce_mean(cross_entropy)


# In[488]:


optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)


# In[489]:


correct_prediction = tf.equal(y_pred_cls, y_true_cls)


# In[490]:


accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# In[491]:


#with tf.device('/gpu:0'):


# In[492]:


session = tf.Session()


# In[493]:


session.run(tf.initialize_all_variables())


# In[494]:


train_batch_size = batch_size


# In[495]:


def print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%}, Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))


# In[496]:


total_iterations = 0

def optimize(num_iterations):
    # Ensure we update the global variable rather than a local copy.
    global total_iterations

    # Start-time used for printing time-usage below.
    start_time = time.time()
    
    best_val_loss = float("inf")
    patience = 0

    for i in range(total_iterations,
                   total_iterations + num_iterations):

        # Get a batch of training examples.
        # x_batch now holds a batch of images and
        # y_true_batch are the true labels for those images.
        x_batch, y_true_batch, _, cls_batch = data.train.next_batch(train_batch_size)
        x_valid_batch, y_valid_batch, _, valid_cls_batch = data.valid.next_batch(train_batch_size)

        # Convert shape from [num examples, rows, columns, depth]
        # to [num examples, flattened image shape]

        x_batch = x_batch.reshape(train_batch_size, img_size_flat)
        x_valid_batch = x_valid_batch.reshape(train_batch_size, img_size_flat)

        # Put the batch into a dict with the proper names
        # for placeholder variables in the TensorFlow graph.
        feed_dict_train = {x: x_batch,
                           y_true: y_true_batch}
        
        feed_dict_validate = {x: x_valid_batch,
                              y_true: y_valid_batch}

        # Run the optimizer using this batch of training data.
        # TensorFlow assigns the variables in feed_dict_train
        # to the placeholder variables and then runs the optimizer.
        session.run(optimizer, feed_dict=feed_dict_train)
        

        # Print status at end of each epoch (defined as full pass through training dataset).
        if i % int(data.train.num_examples/batch_size) == 0: 
            val_loss = session.run(cost, feed_dict=feed_dict_validate)
            epoch = int(i / int(data.train.num_examples/batch_size))
            
            print_progress(epoch, feed_dict_train, feed_dict_validate, val_loss)
            
            if early_stopping:    
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience = 0
                else:
                    patience += 1

                if patience == early_stopping:
                    break

    # Update the total number of iterations performed.
    total_iterations += num_iterations

    # Ending time.
    end_time = time.time()

    # Difference between start and end-times.
    time_dif = end_time - start_time

    # Print the time-usage.
    print("Time elapsed: " + str(timedelta(seconds=int(round(time_dif)))))


# In[497]:


def plot_example_errors(cls_predict, correct):
    
    incorrect = (correct == False)
    
    images = data.valid.images[incorrect]
    
    cls_pred = cls_pred[incorrect]
    
    cls_true = data.valid.cls[incorrect]
    
    plot_images(images=images[0:9],
               cls_true=cls_true[0:9],
               cls_pred=cls_pred[0:9])


# In[498]:


def plot_confusion_matrix(cls_pred):
    
    cls_true = data.valid.cls
    
    cm = confusion_matrix(y_true=cls_true,
                          y_pred=cls_pred)
    print(cm)
    plt.matshow(cm)
    
    plt.colorbar()
    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, range(num_classes))
    plt.yticks(tick_marks, range(num_classes))
    plt.xlabel('Predicted')
    plt.ylabel('True')

   
    plt.show()
    


# In[499]:


def print_validation_accuracy(show_example_errors=False,
                        show_confusion_matrix=False):

    # Number of images in the test-set.
    num_test = len(data.valid.images)

    # Allocate an array for the predicted classes which
    # will be calculated in batches and filled into this array.
    cls_pred = np.zeros(shape=num_test, dtype=np.int)

  

    # The starting index for the next batch is denoted i.
    i = 0

    while i < num_test:
     
        # The ending index for the next batch is denoted j.
            j = min(i + batch_size, num_test)

        # Get the images from the test-set between index i and j.
            images = data.valid.images[i:j, :].reshape(batch_size, img_size_flat)
        

        # Get the associated labels.
            labels = data.valid.labels[i:j, :]

        # Create a feed-dict with these images and labels.
            feed_dict = {x: images,
                         y_true: labels}

        # Calculate the predicted class using TensorFlow.
            cls_pred[i:j] = session.run(y_pred_cls, feed_dict=feed_dict)

        # Set the start-index for the next batch to the
        # end-index of the current batch.
            i = j

    cls_true = np.array(data.valid.cls)
    cls_pred = np.array([classes[x] for x in cls_pred]) 

    # Create a boolean array whether each image is correctly classified.
    correct = (cls_true == cls_pred)

    # Calculate the number of correctly classified images.
    # When summing a boolean array, False means 0 and True means 1.
    correct_sum = correct.sum()

    # Classification accuracy is the number of correctly classified
    # images divided by the total number of images in the test-set.
    acc = float(correct_sum) / num_test

    # Print the accuracy.
    msg = "Accuracy on Test-Set: {0:.1%} ({1} / {2})"
    print(msg.format(acc, correct_sum, num_test))

    # Plot some examples of mis-classifications, if desired.
    if show_example_errors:
        print("Example errors:")
        plot_example_errors(cls_pred=cls_pred, correct=correct)

    # Plot the confusion matrix, if desired.
    if show_confusion_matrix:
        print("Confusion Matrix:")
        plot_confusion_matrix(cls_pred=cls_pred)


# In[500]:


optimize(num_iterations=1)
#print_validation_accuracy()


# In[501]:


optimize(num_iterations=200) 


# In[502]:


#print_validation_accuracy(show_example_errors=True)


# In[503]:


optimize(num_iterations=500)  # We performed 100 iterations above.


# In[504]:


#print_validation_accuracy(show_example_errors=True)


# In[505]:


optimize(num_iterations=1000) # We performed 1000 iterations above.


# In[506]:


#print_validation_accuracy(show_example_errors=True, show_confusion_matrix=True)


# In[507]:


plt.axis('off')

test_onion = cv2.imread('onion.1.jpeg')
test_onion = cv2.resize(test_onion, (img_size, img_size), cv2.INTER_LINEAR) / 255

preview_onion = plt.imshow(test_onion.reshape(img_size, img_size, num_channels))


# In[508]:


test_tomato = cv2.imread('tomato.1.jpeg')
test_tomato = cv2.resize(test_tomato, (img_size, img_size), cv2.INTER_LINEAR) / 255

preview_tomato= plt.imshow(test_tomato.reshape(img_size, img_size, num_channels))


# In[509]:


test_other = cv2.imread('car3.jpeg')
test_other = cv2.resize(test_other, (img_size, img_size), cv2.INTER_LINEAR) / 255

preview_other= plt.imshow(test_other.reshape(img_size, img_size, num_channels))


# In[510]:


def sample_prediction(test_im):
    
    feed_dict_test = {
        x: test_im.reshape(1, img_size_flat),
        #y_true: np.array([[1, 0,2]])
        y_true: np.array([[0,1,2]])
    }

    test_pred = session.run(y_pred_cls, feed_dict=feed_dict_test)
    print("predicted ",test_pred)
    return classes[test_pred[0]]

print("Predicted class for test_ onion : {}".format(sample_prediction(test_onion)))
print("Predicted class for test_ tomato: {}".format(sample_prediction(test_tomato)))
print("Predicted class for test_ other: {}".format(sample_prediction(test_other)))


# In[511]:


import cv2
import numpy as np

cap = cv2.VideoCapture(0)
while True:
    ret,img = cap.read()
    print("read" ,ret)
    cv2.imshow('video output',img)
    temp = cv2.resize(img, (32,32))
    feed_dict_test = {
        x: temp.reshape(1, img_size_flat),
        y_true: np.array([[0, 1, 2]])
    }

    test_pred = session.run(y_pred_cls, feed_dict=feed_dict_test)
    print (test_pred[0])
    if(test_pred[0] == 0):
        print("Predicted image is onion")
    elif(test_pred[0] ==1):
        print("Predicted image is tomato")
    else:
        print("Others images")
        
    k=cv2.waitKey(10) 
    if k == 27:
        break
cap.release()
cv2.destroyAllWindows()

