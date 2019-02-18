import cv2
import tensorflow as tf
import os
import numpy as np
from build_model import model_tools
from config import *

model=model_tools()
model_folder='checkpoints'
image='data/baroque/0.jpg'
img=cv2.imread(image)
session=tf.Session()
img=cv2.resize(img,(100,100))
img=img.reshape(1,100,100,3)
labels = np.zeros((1, number_of_classes))

#Create a saver object to load the model
# saver = tf.train.import_meta_graph(os.path.join(model_folder,'.meta'))
saver = tf.train.import_meta_graph("./checkpoints/checkpoints.meta")

#restore the model from our checkpoints folder
# saver.restore(session,os.path.join(model_folder,'/'))
print("here")
saver.restore(session,tf.train.latest_checkpoint("checkpoints"))


#Create graph object for getting the same network architecture
graph = tf.get_default_graph()

#Get the last layer of the network by it's name which includes all the previous layers too
network = graph.get_tensor_by_name("add_4:0")

#create placeholders to pass the image and get output labels
im_ph= graph.get_tensor_by_name("Placeholder:0")
label_ph = graph.get_tensor_by_name("Placeholder_1:0")

#Inorder to make the output to be either 0 or 1.
network=tf.nn.softmax(network)

# Creating the feed_dict that is required to be fed to calculate y_pred
feed_dict_testing = {im_ph: img, label_ph: labels}
result=session.run(network, feed_dict=feed_dict_testing)
amax = tf.argmax(result, axis = -1)

print(tf.argmax(result, axis = -1))
print(all_classes.index("Baroque"))
with tf.Session():
    print(len(result[0]))
    print(amax.eval())
    print(all_classes[amax.eval()[0]])
