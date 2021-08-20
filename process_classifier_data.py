import os
import json
import numpy as np
import random
#from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform
from skimage.util import img_as_uint #crop
from skimage.color import rgb2gray
from skimage.transform import resize
import pickle
import sys

class ClassifierDataset(Dataset):

    def __init__(self, root_dir, dataset='train_data', transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        label_map = {'(': 0, ')': 1, ',': 2, '.' :3, '-': 4, '/': 5, '<': 6, ' ':7}
        for i in range(10):
            label_map[chr(48+i)] = 8+i
        for i in range(26):
            label_map[chr(65+i)] = 18+i
            label_map[chr(97+i)] = 18+i
        self.root_dir = root_dir 
        pkl_file = open(root_dir, 'rb')
        filenames = pickle.load(pkl_file)
        pkl_file.close()     
        self.transform = transform
        self.sample_list = []
        for _f in filenames[dataset]:
            image = img_as_uint(io.imread(_f[0], as_gray=True))
            with open(_f[1], 'r') as fp:
                label = json.load(fp)
            for item in label:
                if len(item['start_x']) != len(item['end_x']):
                    print(_f[0].split("/")[-1], item['start_x']), len(item['end_x'])
                    continue
                values = [chr(k) for k in item['values']]
                i = 0 
                for val_index in range(len(values)):
                    sample = {}
                    # y_vector = np.zeros(44)
                    if (values[val_index] != ' ' and len(values)!=len(item['start_x'])) or len(values)==len(item['start_x']):
                        img_crop = image[item['let_blines'][i][0]:item['let_blines'][i][1], item['start_x'][i]:item['end_x'][i]]
                        i += 1
                    else:
                        _, y, _, h = item['line_rect']
                        y1 = y #item['let_blines'][i-1][0]
                        y2 = y+h #item['let_blines'][i-1][1]
                        try:
                            if i==0:
                                x1 = item['start_x'][i] - 19
                                x2 = item['start_x'][i]
                            elif i<len(item['start_x']):
                                mid = int((item['end_x'][i-1] + item['start_x'][i])/2)
                                x1 = mid - 9
                                x2 = mid + 10
                            else:
                                x1 = item['end_x'][i-1]
                                x2 = x1+19
                            if x1 >= x2 or y1 == y2:
                                print(i, y1, y2, x1, x2)
                                print(item['start_x'])
                                print(item['end_x'])
                                print(values)
                                sys.exit(1)
                            img_crop = image[y1:y2, x1:x2]
                        except IndexError:
                            print(len(values), len(item['let_blines']), len(item['end_x']), len(item['start_x']), i)
                            print(values)                            
                            sys.exit(1)
                        
                    # y_vector[label_map[values[val_index]]] = 1
                    try:
                        sample['image'] = resize(img_crop, (15,19)).reshape(1, 15, 19)
                        sample['label'] = label_map[values[val_index]]
                        self.sample_list.append(sample)
                    except:
                        print(y1, y2, x1, x2, image.shape)
                    

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.transform:
            for item in range(len(self.sample_list)):
                self.sample_list[item] = self.transform(self.sample_list[item])
        return self.sample_list[idx]

class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        # h and w are swapped for landmarks because for images,
        # x and y axes are axis 1 and 0 respectively
        # landmarks = landmarks * new_w / w #[new_w / w, new_h / h]

        return {'image': img, 'label': label}

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        
        #image = image.transpose((2, 0, 1))
        if not torch.is_tensor(image):
            image = torch.from_numpy(image)
        if not torch.is_tensor(landmarks):
            landmarks = torch.tensor(landmarks)
        return {'image': image,
                'label': landmarks}