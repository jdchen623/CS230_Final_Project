import os, os.path
import shutil


TEST_DIR = "data/test"
VAL_DIR = "data/val"

test_files = [name for name in os.listdir(TEST_DIR) if name.endswith("jpg")]
val_files = [name for name in os.listdir(VAL_DIR) if name.endswith("jpg")]

print("number of files currently in test directory:", len(test_files))
print("number of files currently in val directory:", len(val_files))

for file_index in range(0, int(len(test_files)/2)):
    file_name = test_files[file_index]
    file_path = os.path.join("data/test/", file_name)
    new_path = os.path.join("data/val/", file_name)
    os.rename(file_path, new_path)

test_files = [name for name in os.listdir(TEST_DIR) if name.endswith("jpg")]
val_files = [name for name in os.listdir(VAL_DIR) if name.endswith("jpg")]
print("number of files currently in test directory after split:", len(test_files))
print("number of files currently in val directory after split:", len(val_files))
