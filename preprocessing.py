import cv2
import os
from labelling import makeFileNameAndStylePairs
import math


def seperate_images_by_class():
    print("seperate_images_by_class")
    raw_data = 'raw_data'
    os.makedirs("raw_data")
    unprocessed_data = "unprocessed_data/training_data"
    labels_dict = makeFileNameAndStylePairs()
    #os.walk() returns tuple(dirpath, dirnames, filenames)
    for i in os.walk(unprocessed_data):
        filenames = i[2]
        for fl in filenames:
            if fl == ".DS_Store": continue
            label = labels_dict[fl]
            if type(label) is not str: continue
            path = os.path.join(raw_data,label)
            if not os.path.exists(path):
                os.makedirs(path)
            original_path = os.path.join(unprocessed_data, fl)
            im=cv2.imread(original_path)
            print("path",path)
            cv2.imwrite(os.path.join(path,fl), im)

    # for i in os.walk(raw_data):
    #     path = os.path.join(data_path,class_labels[category_count])


def image_processing(raw_data,data_path,height,width):
    if not os.path.exists(raw_data):
        seperate_images_by_class()
    class_labels=[]
    category_count=0
    for i in os.walk(raw_data):
        #greater than 1 because the directory currently has a .ds_store file
        print(i)
        if len(i[2])>0:
            counter=0
            images=i[2]
            class_name=i[0].strip('\\')
            print("class_name",class_name)
            print(class_labels)
            print(category_count)
            path=os.path.join(data_path,class_labels[category_count])
            for image in images:
                print("class_name + image",class_name+'/'+image)
                im=cv2.imread(class_name+'/'+image)
                im=cv2.resize(im,(height,width))
                if not os.path.exists(path):
                    os.makedirs(path)
                cv2.imwrite(os.path.join(path,str(counter)+'.jpg'),im)
                counter+=1
            category_count+=1
        else:
            number_of_classes=len(i[1])
            print(number_of_classes,i[1])
            class_labels=i[1][:]

if __name__=='__main__':
    height = 100
    width = 100
    raw_data = 'rawdata'
    data_path = 'data'
    if not os.path.exists(data_path):
        image_processing(raw_data, data_path, height, width)
