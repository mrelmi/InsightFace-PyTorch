import os
import pickle

import cv2 as cv
import mxnet as mx
from mxnet import recordio
from tqdm import tqdm

from config import path_imgidx, path_imgrec, IMG_DIR, pickle_file
# from utils import ensure_folder

from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *

import sys 
import re 

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

if __name__ == "__main__":
    # ensure_folder(IMG_DIR)
    # imgrec = recordio.MXIndexedRecordIO(path_imgidx, path_imgrec, 'r')
    # print(len(imgrec))

    samples = []
    class_ids = set()

    # # %% 1 ~ 51795101

    # try:
    #     for i in tqdm(range(10000000)):
    #         # print(i)
    #         header, s = recordio.unpack(imgrec.read_idx(i + 1))
    #         img = mx.image.imdecode(s).asnumpy()
    #         # print(img.shape)
    #         img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
    #         # print(header.label)
    #         # print(type(header.label))
    #         label = int(header.label)
    #         class_ids.add(label)
    #         filename = '{}.jpg'.format(i)
    #         samples.append({'img': filename, 'label': label})
    #         filename = os.path.join(IMG_DIR, filename)
    #         cv.imwrite(filename, img)
    #         # except KeyboardInterrupt:
    #         #     raise
    # except Exception as err:
    #     print(err)

    images_path = scanAllImages(IMG_DIR)
    i = 0
    for path in tqdm(images_path):
      try:
        img = cv.imread(path)
        label = int(os.path.basename(os.path.dirname(path)))
        class_ids.add(label)
        filename = '{}.jpg'.format(i)
        samples.append({'img':filename,'label':label})
        
        filename = os.path.join(IMG_DIR, filename)
        os.remove(path)
        cv.imwrite(filename, img)

        i += 1
      except:
        continue
    with open(pickle_file, 'wb') as file:
        pickle.dump(samples, file)



    print('num_samples: ' + str(len(samples)))

    class_ids = list(class_ids)
    print(len(class_ids))
    print(max(class_ids))
