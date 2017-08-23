# assuming a csv file with a name in column 0 and the image url in column 1
import requests
import numpy as np
from multiprocessing import Pool
from glob import glob
import os

root = '/Users/luzan/data/WTBI/images'

with open('images.csv','r') as f:
    csv = f.readlines()
    csv = [o.strip() for o in csv]
    csv = [(o[:9],o[10:]) for o in csv]

def fetch(line):
    idx, url = line
    r = requests.get(url, allow_redirects=True)
    if len(r.content) == 0:
        print('{} no pic.'.format(idx))
        return
    with open(os.path.join(root,'{}.png'.format(idx)), 'wb') as f:
        f.write(r.content)
    print('{} done.'.format(idx))

if __name__ == '__main__':
    e = glob(os.path.join(root, '*.png'))
    e = set( [int(os.path.basename(o)[:9]) for o in e])
    m = set(range(1,424840))
    m = m.difference(e)
    m = np.array(sorted(list(m))) - 1
    csv = [csv[i] for i in m]
    import random
    random.shuffle(csv)
    # print(csv)
    p = Pool(8)
    p.map(fetch, csv)
