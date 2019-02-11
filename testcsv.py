#import tensorflow as tf


#file_queue = tf.train.string_input_producer(['/data/all_data_info.csv'])
#reader = tf.TextLineReader(skip_header_lines=1)
#_, rows = reader.read_up_to(file_queue,num_records = 100)
#expanded_rows = tf.expand_dims(rows, axis=-1)
#record_defaults = [[0] for _ in range(28*28+1)]
#columns = tf.decode_csv(expanded_rows,record_defaults=record_defaults)
#print(columns[0])



import pandas

def makeFileNameAndStylePairs():
    
    #use pandas to load CSV data into filenames and styles
    colnames = ["new_filename", "style_type"]
    data = pandas.read_csv('data/all_data_info.csv', usecols=colnames)
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
        dict[filename] = index


    return dict


print(makeFileNameAndStylePairs())
