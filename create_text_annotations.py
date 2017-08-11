import numpy as np

import os
import json
import io
import PIL.Image

root = '/Users/luzan/data/WTBI/meta/json/'
img_path_template = '/Users/luzan/data/WTBI/images/{:09}.png'
annotations_dir = '/Users/luzan/data/WTBI/annotations'

classes = ['bags', 'belts', 'dresses', 
           'eyewear', 'footwear', 'hats', 
           'leggings', 'outerwear', 'pants', 
           'skirts', 'tops']
name2cls = { v:k+1 for k,v in enumerate(classes) }

import sys
import cv2

annotations = {}
img_sizes = {}
for dataset in ['train', 'test']:
    template = os.path.join(root, dataset+'_pairs_{}.json')
    for clsname in classes:
        with open(template.format(clsname)) as f:
            a = json.load(f)

        for d in a:
            k = d['photo']

            if k not in img_sizes:
                full_path = img_path_template.format(k)
                im=PIL.Image.open(full_path)
                height = im.height
                width = im.width
                img_sizes[k] = (height,width)
            else:
                height, width = img_sizes[k]

            # fix for bad box
            b = d['bbox']
            if b['height']*b['width'] == 0: continue

            top, left, bheight, bwidth = b['top'],b['left'],b['height'],b['width']
            xmin, ymin = left, top
            xmax, ymax = min(xmin + bwidth - 1, width), min(ymin + bheight - 1, height)
            assert xmax-xmin > 0 and ymax - ymin > 0

            if k not in annotations:
                annotations[k]=[]
            annotations[k].append([name2cls[clsname], xmin, ymin, xmax, ymax])

# annotations for each pic are a text file where each instance is annotated
# <class> <xmin> <ymin> <xmax> <ymax> where class id's start at 1
# (0 is background)
def main():
    for idx, annos in annotations.items():
        with open(os.path.join(annotations_dir, '{:09}.txt'.format(idx)), 'w') as f:
            for anno in annos:
                f.write('{} {} {} {} {}\n'.format(*anno))

if __name__=='__main__':
    main()
