
import os
import sys
sys.path.append("/home/juwon/tmp/cocoapi/PythonAPI")
from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import pylab
import matplotlib.pyplot as plt
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

dataDir='/home/share/data/coco'
dataType='train2017'
annFile='{}/annotations/instances_{}.json'.format(dataDir,dataType)
coco=COCO(annFile)
cats = coco.loadCats(coco.getCatIds())
nms=[cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms)))

nms = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms)))

catIds = coco.getCatIds(catNms=['cat','dog']);
catIds = coco.getCatIds(catNms=[i for i in range(80)]);
imgIds = coco.getImgIds(catIds=catIds );
imgIds = coco.getImgIds(imgIds = [324158])
img = coco.loadImgs(imgIds[np.random.randint(0,len(imgIds))])[0]


annFile = '{}/annotations/captions_{}.json'.format(dataDir,dataType)
coco_caps=COCO(annFile)

I = io.imread(img['coco_url'])
annIds = coco_caps.getAnnIds(imgIds=img['id']);
anns = coco_caps.loadAnns(annIds)
coco_caps.showAnns(anns)
plt.imshow(I); plt.axis('off'); plt.show()