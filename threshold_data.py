
import os
import pandas
from collections import OrderedDict

NUM_IMAGES_THRESHOLD = 500
DIR_PATH = "data/train/"
LABELS_FILE = "train_labels.txt"

def makeFileNameAndStylePairsCsv():

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
    style_dict = {}
    for key, value in dict.items():
        if value in style_dict:
            style_dict[value] += 1
        else:
            style_dict[value] = 1

    dd = OrderedDict(sorted(style_dict.items(), key=lambda x: -x[1]))

    return dict, style_dict

def meets_threshold(style_dict, count_dict, filename):
    if filename in style_dict:
        style = style_dict[filename]
        if style in count_dict:
            return count_dict[style] > NUM_IMAGES_THRESHOLD
        else:
            assert("Missing style")
    else:
        assert("Missing filename")

def make_label_txt_file(style_dict, label_dict, data_path):
    filepath = LABELS_FILE
    with open(filepath, 'a') as label_file:
        for filename in os.listdir(data_path):
            if filename in style_dict:
                file_style = style_dict[filename]
                if file_style in label_dict:
                    line = filename + " " + str(label_dict[file_style]) + "\n"
                    label_file.write(line)
                else:
                    assert("Missing style")
            else:
                assert("Missing file")

def remove_nonthreshold_files(style_dict, count_dict):
    files_to_delete = []
    for filename in os.listdir(DIR_PATH):
        if filename.endswith(".jpg"):
            if not meets_threshold(style_dict, count_dict, filename):
                files_to_delete.append(filename)

    for filename in files_to_delete:
        filepath = DIR_PATH  + filename
        os.remove(filepath)

def get_label_dict(style_dict, data_path):
    styles = []
    for filename in os.listdir(data_path):
        if filename in style_dict and style_dict[filename] not in styles and style_dict[filename] == style_dict[filename] :
            styles.append(style_dict[filename])

    styles = sorted(styles)
    print("styles:",len(styles))
    label_dict = {}
    label = 0
    for style in styles:
        label_dict[style] = label
        label += 1
    return label_dict

def remove_nan(style_dict, dir_path):
    for filename in os.listdir(dir_path):
        if filename in style_dict and style_dict[filename] and style_dict[filename] != style_dict[filename]:
            file_to_delete = dir_path + filename
            os.remove(file_to_delete)




style_dict, count_dict = makeFileNameAndStylePairsCsv()
# remove_nan(style_dict, DIR_PATH)
#
label_dict = get_label_dict(style_dict, DIR_PATH)
# make_label_txt_file(style_dict, label_dict, DIR_PATH)
