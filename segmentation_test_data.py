import os
import json
import numpy as np
import random
#from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import io, transform
#from skimage.util import crop
from skimage.color import rgb2gray
from skimage.transform import resize
import pickle

class SegmentaionTest(Dataset):

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir        
        self.transform = transform
        self.sample_list = []
        char_i= 0
        pkl_file = open(self.root_dir, 'rb')
        image_list = pickle.load(pkl_file)
        pkl_file.close()
        ind = 0
        word_list_dict = {}
        for xml_file in image_list:
            for reg in image_list[xml_file]:
                line_ = 0
                for line in image_list[xml_file][reg]:
                    word_ = 0
                    for word in line:
                        # if word.shape[-1] <400:
                        #     word = np.pad(word, ((0,0), (int((800-word.shape[-1])/2), int((800-word.shape[-1])/2))), 'maximum')
                        sample = {}
                        sample['image'] = resize(word, (33,800)).reshape(1,33,800)
                        sample['landmark'] = np.zeros((1,1,800))
                        sample['values'] = ind
                        word_list_dict[ind] = [xml_file, reg, line_, word_]
                        word_ +=1
                        self.sample_list.append(sample)
                    line_ += 1
                
        output = open('word_list_dict.pkl', 'wb')
        pickle.dump(word_list_dict, output)
        output.close()

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.transform:
            for item in range(len(self.sample_list)):
                self.sample_list[item] = self.transform(self.sample_list[item])
        return self.sample_list[idx]



class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, landmarks, val = sample['image'], sample['landmark'], sample['values']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        
        #image = image.transpose((2, 0, 1))
        if not torch.is_tensor(image):
            image = torch.from_numpy(image)
        if not torch.is_tensor(landmarks):
            landmarks = torch.from_numpy(landmarks)
        return {'image': image,
                'landmark': landmarks, 'values': val}