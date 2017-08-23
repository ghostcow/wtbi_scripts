
import numpy as np
import os
import json
import io
import PIL.Image
import sys
import cv2


root = '/Users/luzan/data/WTBI/meta/json/' # path to json metadata
img_path_template = '/Users/luzan/data/WTBI/images/{:09}.png' # path to images

# object classes start at 1, and 0 is background
classes = ['bags', 'belts', 'dresses',
           'eyewear', 'footwear', 'hats',
           'leggings', 'outerwear', 'pants',
           'skirts', 'tops']
name2cls = { v:k+1 for k,v in enumerate(classes) }


annotations = {}
img_sizes = {}
for dataset in ['train', 'test']:
    template = os.path.join(root, dataset+'_pairs_{}.json')
    for clsname in classes:
        with open(template.format(clsname)) as f:
            a = json.load(f)

        for d in a:
            k = d['photo']

            # get size of image
            if k not in img_sizes:
                full_path = img_path_template.format(k)
                im = PIL.Image.open(full_path)
                height = im.height
                width = im.width
                img_sizes[k] = (height,width)
            else:
                height, width = img_sizes[k]

            # fix for bad boxes
            b = d['bbox']
            if b['height']*b['width'] == 0: continue

            top, left, bheight, bwidth = b['top'],b['left'],b['height'],b['width']
            xmin, ymin = left, top
            xmax, ymax = min(xmin + bwidth - 1, width), min(ymin + bheight - 1, height)
            assert xmax-xmin > 0 and ymax-ymin > 0

            if k not in annotations:
                annotations[k]=[]
            annotations[k].append([name2cls[clsname], xmin, ymin, xmax, ymax])


def main():
    anno_dir = '/Users/luzan/data/WTBI/annotations'
    if not os.path.exists(anno_dir):
        os.makedirs(anno_dir)
    # each image gets a text file containing all bounding boxes in image
    # one bounding box per line, the format: <CLASS> <XMIN> <YMIN> <XMAX> <YMAX>
    for idx, annos in annotations.items():
        with open(os.path.join(anno_dir, '{:09}.txt'.format(idx)), 'w') as f:
            for anno in annos:
                f.write('{} {} {} {} {}\n'.format(*anno))

if __name__=='__main__':
    main()
