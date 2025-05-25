import os
import shutil
script_dir = os.path.dirname(os.path.abspath(__file__))

print(script_dir)

paths = os.path.join(script_dir, 'Dataset', 'train', 'combined_train')

new_path = os.path.join(script_dir, 'Dataset', 'yolo_dataset','labels', 'combined_train')
"""
# Now you can list the directory
for i in os.listdir(paths):
    if i.endswith(".txt"):
        shutil.move(os.path.join(paths,i), new_path)
        #os.remove(os.path.join(paths,i))"""