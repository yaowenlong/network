from xml.etree import ElementTree as ET
import os
import torch.utils
import torch
from torch.utils.data.dataloader import DataLoader
from PIL import Image
import numpy as np
import torch.tensor as Tensor
class MydataSet(torch.utils.data.Dataset):

    def __init__(self, xmlpath, imgpath, transform=None, target_transform=None):
        s = []
        imgs = []
        files = os.listdir(xmlpath)
        dict = {}

        self._classes = ('__background__' , # always index 0
                         'aeroplane', 'bicycle', 'bird', 'boat',
                         'bottle', 'bus', 'car', 'cat', 'chair',
                         'cow', 'diningtable', 'dog', 'horse',
                         'motorbike', 'person', 'pottedplant',
                         'sheep', 'sofa', 'train', 'tvmonitor')


        for file in files:
            input = ''
            label = []
            a = torch.tensor(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),dtype=torch.float)
            if not os.path.isdir(file):
                tree = ET.parse(xmlpath + "/" + file)
                root = tree.getroot()

                for child in root:
                    if child.tag == "filename":
                        input = child.text
                    if child.tag == "object":
                        for i in child :
                            if i.tag == "name":
                                label.append(i.text)
            if len(label) < 2:
                s = label[0]
                label.append(s)

            a[self._classes.index(label[0])] = 1
            #print(a)
           # print(a[self._classes.index(label[0])])
            imgs.append((input, a))
            self.imgs = imgs
            self.transform = transform
            self.target_transform = target_transform
            self.imgpath = imgpath
    def __getitem__(self, index):

        fn, label = self.imgs[index]
        img = Image.open(self.imgpath + "/" + fn)


        if self.transform is not None:
            img = self.transform(img)

            np.transpose(img, (1, 2, 0))
        return img, label

    def __len__(self):
        return len(self.imgs)
    @staticmethod
    def collate_fn(batch):
        batch.sort(key=lambda x: len(x[1]), reverse=True)
        img, label = zip(*batch)
        pad_label = []
        lens = []
        max_len = len(label[0])
        for i in range(len(label)):
            temp_label = [0] * max_len
            temp_label[:len(label[i])] = label[i]
            pad_label.append(temp_label)
            lens.append(len(label[i]))
        return img, pad_label, lens







