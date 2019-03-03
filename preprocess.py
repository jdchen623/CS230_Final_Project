
import os
import math

TRAIN_PATH = "data/train"
VAL_PATH = "data/val"
TEST_PATH = "data/test"

def seperate_images_by_class(path):
    print("seperate_images_by_class")
    unprocessed_path = path
    labels_dict = makeFileNameAndStylePairs()
    #os.walk() returns tuple(dirpath, dirnames, filenames)
    for i in os.walk(unprocessed_path):
        filenames = i[2]
        for fl in filenames:
            if fl == ".DS_Store": continue
            label = labels_dict[fl]
            if type(label) is not str: continue
            original_path = os.path.join(unprocessed_path, fl)
            new_dir = os.path.join(unprocessed_path, label)
            new_path = os.path.join(new_dir,fl)
            if not os.path.exists(new_dir):
                os.makedirs(new_dir)
            os.rename(original_path, new_path)
            print(original_path, " moved to ", new_path)

import pandas

def makeFileNameAndStylePairs():

    #use pandas to load CSV data into filenames and styles
    colnames = ["new_filename", "style_type"]
    data = pandas.read_csv('all_data_info.csv', usecols=colnames)
    filenames = data.new_filename
    styles = data.style_type

    #extract all style types to enumerate
    seen_styles = []
    style_index = []
    for sty in styles:
        if sty not in seen_styles:
            seen_styles.append(sty)
            style_index.append(sty)

    #place all filenames and styles into a dictionary
    dict = {}
    for i in range(len(styles)):
        sty = styles[i]
        filename = filenames[i]
        index = style_index.index(sty)
        # dict[filename] = index
        dict[filename] = sty


    return dict

def get_validation_images():
    labels_dict = makeFileNameAndStylePairs()
    validation_file_names = []
    validation_file_labels = []


    for root, dirs, files in os.walk("data/validation"):
        for directory in dirs:
            for subroot, subdirs, subfiles in os.walk(directory):
                if subfile == ".DS_Store": continue
                validation_file_names.append(os.path.join(subroot, subdir, subfile))
                validation_file_labels.append(labels_dict[subfile])
    # print(validation_file_labels)
#seperate_images_by_class(TRAIN_PATH)
#seperate_images_by_class(VAL_PATH)
#seperate_images_by_class(TEST_PATH)

get_validation_images()
