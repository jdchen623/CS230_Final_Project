
import os
import pandas
from collections import OrderedDict

NUM_IMAGES_THRESHOLD = 1000

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
    print(len(files_to_delete))


style_dict, count_dict = makeFileNameAndStylePairs()

files_to_delete = []
for filename in os.listdir("data/test"):
    if filename.endswith(".jpg"):
        if not meets_threshold(style_dict, count_dict, filename):
            files_to_delete.append(filename)

for filename in files_to_delete:
    filepath = "data/test/" + filename
    os.remove(filepath)
