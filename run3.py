import cv2
import numpy as np
import glob


class_paths = glob.glob('dataset/training/*')
class_names = [name[17:] for name in class_paths]
print(class_paths)
print(class_names)
