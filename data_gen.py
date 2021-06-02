import os
import pickle

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import IMG_DIR
from config import pickle_file



from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

import sys 
import re 
# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}


class ArcFaceDataset(Dataset):
    def __init__(self, split):
        with open(pickle_file, 'rb') as file:
            data = pickle.load(file)

        self.split = split
        self.samples = data
        self.transformer = data_transforms['train']

    def __getitem__(self, i):
        sample = self.samples[i]
        filename = sample['img']
        label = sample['label']

        filename = os.path.join(IMG_DIR, filename)
        img = Image.open(filename)
        img = self.transformer(img)

        return img, label

    def __len__(self):
        return len(self.samples)
import os
import pickle

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from config import IMG_DIR
from config import pickle_file

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.125, contrast=0.125, saturation=0.125),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}
DEFAULT_ENCODING = 'utf-8'
def ustr(x):
    '''py2/py3 unicode helper'''

    if sys.version_info < (3, 0, 0):
        from PyQt4.QtCore import QString
        if type(x) == str:
            return x.decode(DEFAULT_ENCODING)
        if type(x) == QString:
            #https://blog.csdn.net/friendan/article/details/51088476
            #https://blog.csdn.net/xxm524/article/details/74937308
            return unicode(x.toUtf8(), DEFAULT_ENCODING, 'ignore')
        return x
    else:
        return x

def natural_sort(list, key=lambda s: s):
  """
  Sort the list into natural alphanumeric order.
  """

  def get_alphanum_key_func(key):
    convert = lambda text: int(text) if text.isdigit() else text
    return lambda s: [convert(c) for c in re.split('([0-9]+)', key(s))]

  sort_key = get_alphanum_key_func(key)
  list.sort(key=sort_key)


def scanAllImages(folderPath):
  extensions = ['.%s' % fmt.data().decode("ascii").lower() for fmt in QImageReader.supportedImageFormats()]
  images = []

  for root, dirs, files in os.walk(folderPath):
    for file in files:
      if file.lower().endswith(tuple(extensions)):
        relativePath = os.path.join(root, file)
        path = ustr(os.path.abspath(relativePath))
        images.append(path)
  natural_sort(images, key=lambda x: x.lower())
  return images

class ArcFaceDataset(Dataset):
    def __init__(self, split):
        # with open(pickle_file, 'rb') as file:
        #     data = pickle.load(file)

        self.split = split
        # self.samples = data
        self.transformer = data_transforms['train']


        self.images_path = scanAllImages(IMG_DIR)
    def __getitem__(self, i):
        
        path = self.images_path[i]
        # filename = sample['img']
        filename = path
        label = int(os.path.basename(os.path.dirname(path)))

        # filename = os.path.join(IMG_DIR, filename)
        img = Image.open(filename)
        img = self.transformer(img)

        return img, label

    def __len__(self):
        return len(self.images_path)
