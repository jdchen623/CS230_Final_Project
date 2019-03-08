import os, random

#should use this on Impressionism, Realism, Romanticism, and Expressionism
directory_name = "data/train/Expressionism"
max_files = 5000

path,dirs,files = next(os.walk(directory_name))
file_count = len(files)
print(file_count)

while (file_count > max_files):
    os.remove(directory_name + "/" + str(random.choice(os.listdir(directory_name))))
    file_count -= 1

print(file_count)
