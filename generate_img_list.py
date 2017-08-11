import numpy as np

import os
import json
import io
import PIL.Image

root = '/Users/luzan/data/WTBI/meta/json/'


classes = ['bags', 'belts', 'dresses', 
           'eyewear', 'footwear', 'hats', 
           'leggings', 'outerwear', 'pants', 
           'skirts', 'tops']
name2cls = { v:k+1 for k,v in enumerate(classes) }

import sys
import tensorflow as tf
import cv2

key_template = '{:09}'
keys = {}
for dataset in ['train', 'test']:
    keys[dataset] = set()
    template = os.path.join(root, dataset+'_pairs_{}.json')
    for clsname in classes:
        with open(template.format(clsname)) as f:
            a = json.load(f)

        for d in a:
            k = d['photo']
            keys[dataset].add(key_template.format(k))


for d in ['train', 'test']:
    with open('/Users/luzan/data/WTBI/{}.txt'.format(d), 'w') as f:
        for k in keys[d]:
            f.write('{}\n'.format(k))
