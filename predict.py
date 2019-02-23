import cv2
import tensorflow as tf
import os
import numpy as np
from build_model import model_tools
from config import *
from labelling import makeFileNameAndStylePairs
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import f1_score
import gc

NUM_TEST_FILES = 100
model=model_tools()
model_folder='checkpoints'

session=tf.Session()
#Create a saver object to load the model
# saver = tf.train.import_meta_graph(os.path.join(model_folder,'.meta'))
saver = tf.train.import_meta_graph("./checkpoints/checkpoints.meta")

#restore the model from our checkpoints folder
# saver.restore(session,os.path.join(model_folder,'/'))
saver.restore(session,tf.train.latest_checkpoint("checkpoints"))

#Create graph object for getting the same network architecture
graph = tf.get_default_graph()

def predict(image_path):

    image=image_path
    img=cv2.imread(image)
    img=cv2.resize(img,(100,100))
    img=img.reshape(1,100,100,3)
    labels = np.zeros((1, number_of_classes))

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

    with tf.Session():
        return all_classes[amax.eval()[0]]

def run_test_set():
    gc.disable()
    labels_dict = makeFileNameAndStylePairs()
    predictions = []
    labels = []
    #creates a dictionary that will be save results of testing
    output_dictionary = {}
    count = 0
    for root, dirs, files in os.walk("unprocessed_data/test_data"):
        for name in files:
            num_files = len(files)
            current_file_index = files.index(name)
            print("%s of %s test files" % (current_file_index, num_files))
            img_path = os.path.join(root,name)
            print(img_path)
            prediction = predict(img_path)
            label = labels_dict[name]
            predictions.append(prediction)
            labels.append(label)
            output_dictionary[img_path]= (label,prediction)
            count += 1
            if count >= NUM_TEST_FILES:
                break
    # print(labels)
    # print(predictions)
    predictions = np.asarray(list(predictions))
    labels = np.asarray(list(labels))

    #f1 score returns (precision, recall, fbeta, )

    precision_and_recall = precision_recall_fscore_support(labels, predictions, average = 'macro')
    f1score = f1_score(labels, predictions, average = 'micro')
    print("preicison and recall", precision_and_recall)
    print("precision: %s, recall: %s"% (precision_and_recall[0], precision_and_recall[1]))
    print("f1 score:", f1score)
    prediction_file = open("saved_results/predict_output.txt", "w")
    # read string from file and call eval() to get back dict
    prediction_file.write(str(output_dictionary))



if __name__=='__main__':
    predict('data/baroque/0.jpg')
    run_test_set()
