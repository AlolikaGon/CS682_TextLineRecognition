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

class SegmentaionMidv500(Dataset):

    def __init__(self, root_dir = 'datasets/midv500-textline/', process = 'resize', transform=None): #process -> resize or append
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir        
        self.transform = transform
        self.sample_list = []
        with open(self.root_dir + 'ground_truth.json', 'r') as fp:
            ground_truth = json.load(fp)
        for image_name in os.listdir(self.root_dir):
            if image_name[-3:] == 'npy':
                sample = {}
                image_np = np.load(self.root_dir + image_name)
                if process == 'resize':
                    im_resize = resize(image_np, (33,700))
                    sample['image'] = im_resize.reshape((1, 33, 700))
                    sample['value'] = ground_truth[image_name]
                    sample['landmark'] = -1
                    self.sample_list.append(sample)
                if process == 'append':
                    im2 = resize(image_np, (33, image_np.shape[1]*33//image_np .shape[0]))
                    im3 = np.zeros((33,(700//im2.shape[1])*im2.shape[1]))
                    ind = 0
                    for i in range(0, 700//im2.shape[1]):
                        im3[:, ind:ind+im2.shape[1]] = im2
                        ind += im2.shape[1]
                    im3 = resize(im3, (33,700))
                    sample['image'] = im3.reshape((1,33,700))
                    sample['value'] = ground_truth[image_name]
                    sample['landmark'] = 700//im2.shape[1]
                    self.sample_list.append(sample)


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
        image, landmarks, val = sample['image'], sample['landmark'], sample['value']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        
        #image = image.transpose((2, 0, 1))
        if not torch.is_tensor(image):
            image = torch.from_numpy(image)
        if not torch.is_tensor(landmarks):
            landmarks = torch.tensor(landmarks)
        return {'image': image,
                'landmark': landmarks, 'value': val}