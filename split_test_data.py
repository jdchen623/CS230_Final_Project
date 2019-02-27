import os, os.path
import shutil


TEST_DIR = "data/test"
VAL_DIR = "data/val"

test_files = [name for name in os.listdir(TEST_DIR) if os.path.isfile(name)]
val_files = [name for name in os.listdir(VAL_DIR) if os.path.isfile(name)]
print("number of files currently in test directory:". len(test_files))
print("number of files currently in val directory:". len(val_files))

for file_index in len(test_files)/2:
    file_name = test_files[file_index]
    file_path = os.join("test/", file_name)
    new_path = os.join("val/", file_name)
    os.rename(file__path, new_path)

test_files = [name for name in os.listdir(TEST_DIR) if os.path.isfile(name)]
val_files = [name for name in os.listdir(VAL_DIR) if os.path.isfile(name)]
print("number of files currently in test directory after split:". len(test_files))
print("number of files currently in val directory after split:". len(val_files))
