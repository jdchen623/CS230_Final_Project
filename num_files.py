import os

DIR_PATH = 'data/train/'

dirs_to_files = {}
for d in os.listdir(DIR_PATH):
    if d == '.DS_Store':
        continue
    count = 0
    for f in os.listdir(DIR_PATH + d):
        if f == '.DS_Store':
            continue
        count += 1
    dirs_to_files[d] = count
print(sorted(dirs_to_files.items(), key=lambda x: x[1], reverse=True))
