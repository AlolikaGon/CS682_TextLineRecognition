import os
import json
import numpy as np
import random
#from PIL import Image
# import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform
#from skimage.util import crop
from skimage.color import rgb2gray
from skimage.transform import resize
import pickle

root_dir = './datasets/train_data_icdar2019/strings/'
image_name_list = []
label_name_list = []
sample_list = []
# print("list dir ", os.listdir(root_dir))
for d in os.listdir(root_dir):
    if len(os.listdir(root_dir + d + '/images')) != 0:
        image_name_list.extend([root_dir + d + '/images/' + name for name in  os.listdir(root_dir + d + '/images')])
        label_name_list.extend([root_dir + d + '/markup/' + name + '.json' for name in  os.listdir(root_dir + d + '/images')])
for i in range(len(image_name_list)):
    with open(label_name_list[i], 'r') as fp:
        label = json.load(fp)
    image = io.imread(image_name_list[i])
    image = rgb2gray(image)
    sample = {}
    for item in label:
        x, y, w, h = item['line_rect']
        img_crop = image[y:y+h, x:x+w] #crop(image, ((y, y+h), (x, x+w)))
        sample['image'] = resize(img_crop, (33, 800)).reshape(1,33,800)
        index = np.asarray(item['cuts_x']) * 800/image.shape[1]
        x_cuts = np.zeros(800)
        x_cuts[index.astype(int)] = 1
        sample['landmark'] = x_cuts.reshape((1,1,800))
        char_list = [chr(val) for val in item['values']]
        sample['values'] = char_list
        sample_list.append(sample)

random.shuffle(sample_list)
output = open('sample_list.pkl', 'wb')
pickle.dump(sample_list, output)
output.close()