
import tensorflow as tf

from object_detection.utils import dataset_util


flags = tf.app.flags
# flags.DEFINE_string('output_path', '/Users/luzan/fashion-proj/wtbi_train.record', 'Path to output TFRecord')
flags.DEFINE_string('output_path', '', 'Path to output TFRecord')
FLAGS = flags.FLAGS


import os
import json
import io
import PIL.Image

root = '/Users/luzan/data/WTBI/meta/json/'
img_path_template = '/Users/luzan/data/WTBI/images/{:09}.png'
# template = os.path.join(root, 'train_pairs_{}.json')
template = os.path.join(root, 'test_pairs_{}.json')

classes = ['bags', 'belts', 'dresses', 
           'eyewear', 'footwear', 'hats', 
           'leggings', 'outerwear', 'pants', 
           'skirts', 'tops']
name2cls = { v:k+1 for k,v in enumerate(classes) }

photos = {}
for clsname in classes:
    with open(template.format(clsname)) as f:
        a = json.load(f)
    
    for d in a:
        k = d['photo']
        # fix for bad box
        if k==8288 and clsname=='bags':
            continue
        if k not in photos:
            photos[k]=[]
        photos[d['photo']].append({'name': clsname, 'bndbox': d['bbox']})
        # print(d['bbox'])

def bbox_transform(b, height, width):
    top, left, bheight, bwidth = b['top'],b['left'],b['height'],b['width']
    xmin, ymin = left, top
    xmax, ymax = min(xmin + bwidth - 1, width), min(ymin + bheight - 1, height)
    assert xmax-xmin>0 and ymax-ymin>0
    retval = [xmin/width, xmax/width, ymin/height, ymax/height]
    return retval

def create_tf_example(idx, data):
    full_path = img_path_template.format(idx)
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    
    height = image.height
    width = image.width
    filename = os.path.basename(full_path).encode()
    encoded_image_data = encoded_jpg
    image_format = b'jpeg'

    boxes=[]
    for d in data:
        try:
            boxes.append(bbox_transform( d['bndbox'], height, width ))
        except:
            print(d)
            raise Exception
    # boxes = [bbox_transform(d['bndbox']) for d in data]
    xmins = [b[0] for b in boxes]
    xmaxs = [b[1] for b in boxes]
    ymins = [b[2] for b in boxes]
    ymaxs = [b[3] for b in boxes]

    classes_text = [d['name'] for d in data]
    classes = [name2cls[o] for o in classes_text]

    tf_example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(filename),
      'image/source_id': dataset_util.bytes_feature(filename),
      'image/encoded': dataset_util.bytes_feature(encoded_image_data),
      'image/format': dataset_util.bytes_feature(image_format),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
      'image/object/class/text': dataset_util.bytes_list_feature([o.encode() for o in classes_text]),
      'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))

    return tf_example


# def create_tf_example(example):
#   # TODO(user): Populate the following variables from your example.
#   height = None # Image height
#   width = None # Image width
#   filename = None # Filename of the image. Empty if image is not from file
#   encoded_image_data = None # Encoded image bytes
#   image_format = None # b'jpeg' or b'png'

#   xmins = [] # List of normalized left x coordinates in bounding box (1 per box)
#   xmaxs = [] # List of normalized right x coordinates in bounding box
#              # (1 per box)
#   ymins = [] # List of normalized top y coordinates in bounding box (1 per box)
#   ymaxs = [] # List of normalized bottom y coordinates in bounding box
#              # (1 per box)
#   classes_text = [] # List of string class name of bounding box (1 per box)
#   classes = [] # List of integer class id of bounding box (1 per box)

#   tf_example = tf.train.Example(features=tf.train.Features(feature={
#       'image/height': dataset_util.int64_feature(height),
#       'image/width': dataset_util.int64_feature(width),
#       'image/filename': dataset_util.bytes_feature(filename),
#       'image/source_id': dataset_util.bytes_feature(filename),
#       'image/encoded': dataset_util.bytes_feature(encoded_image_data),
#       'image/format': dataset_util.bytes_feature(image_format),
#       'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
#       'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
#       'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
#       'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
#       'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
#       'image/object/class/label': dataset_util.int64_list_feature(classes),
#   }))
#   return tf_example

def main():
    writer = tf.python_io.TFRecordWriter(FLAGS.output_path)
    
    for idx, data in photos.items():
        tf_example = create_tf_example(idx, data)
        writer.write(tf_example.SerializeToString())
        print('wrote example {}'.format(idx))
    writer.close()

# def main(_):
#   writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

#   # TODO(user): Write code to read in your dataset to examples variable
    

#   for example in examples:
#     tf_example = create_tf_example(example)
#     writer.write(tf_example.SerializeToString())

#   writer.close()


if __name__=='__main__':
    main()
