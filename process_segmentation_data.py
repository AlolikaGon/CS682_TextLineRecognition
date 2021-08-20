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

class SegmentaionDataset(Dataset):

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir        
        self.transform = transform
        self.image_name_list = []
        self.label_name_list = []
        self.sample_list = []
        char_list_dict = {}
        char_i= 0
        for d in os.listdir(self.root_dir):
            if len(os.listdir(self.root_dir + d + '/images')) != 0:
                self.image_name_list.extend([self.root_dir + d + '/images/' + name for name in  os.listdir(self.root_dir + d + '/images')])
                self.label_name_list.extend([self.root_dir + d + '/markup/' + name + '.json' for name in  os.listdir(self.root_dir + d + '/images')])
        for i in range(len(self.image_name_list)):
            with open(self.label_name_list[i], 'r') as fp:
                label = json.load(fp)
            #image = Image.open(self.image_name_list[i])
            image = io.imread(self.image_name_list[i])
            image = rgb2gray(image)
            sample = {}
            for item in label:
                x, y, w, h = item['line_rect']
                img_crop = image[y:y+h, x:x+w] #crop(image, ((y, y+h), (x, x+w)))
                # sample['image'] = resize(img_crop, (33, 800)).reshape(1,33,800)
                # index = np.asarray(item['cuts_x']) * 800/image.shape[1]
                # x_cuts = np.zeros(800)
                # x_cuts[index.astype(int)] = 1
                # sample['landmark'] = x_cuts.reshape((1,1,800))
                sample['image'] = resize(img_crop, (33,700)).reshape(1,33,700)
                index = (np.asarray(item['cuts_x']) - x) * 699/w
                x_cuts = np.zeros(700)
                x_cuts[index.astype(int)] = 1
                sample['landmark'] = x_cuts.reshape((1,1,700))
                char_list = [chr(val) for val in item['values']]
                sample['values'] = char_i
                char_list_dict[char_i] = char_list
                char_i += 1
                self.sample_list.append(sample)
                
        random.shuffle(self.sample_list)
        output = open('char_list_dict.pkl', 'wb')
        pickle.dump(char_list_dict, output)
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
        image, landmarks = sample['image'], sample['landmark']

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
        landmarks = landmarks * new_w / w #[new_w / w, new_h / h]

        return {'image': img, 'landmark': landmarks}

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