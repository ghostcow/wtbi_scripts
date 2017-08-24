output_path = '/Users/luzan/caffe/data/wtbi/labelmap_wtbi.prototxt'

# generate label file
zerocls = '''item {
  name: "none_of_the_above"
  label: 0
  display_name: "background"
}
'''

classes = ['bags', 'belts', 'dresses',
           'eyewear', 'footwear', 'hats',
           'leggings', 'outerwear', 'pants',
           'skirts', 'tops']

fmt='''item {{
  name: "{clsname}"
  label: {clsid}
  display_name: "{clsname}"
}}
'''

s = zerocls
for i,c in enumerate(classes):
    s += fmt.format(clsname=c, clsid=i+1)

with open(output_path, 'w') as f:
    f.write(s)
