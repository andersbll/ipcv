#!/usr/bin/env python

import scipy as sp

from ipcv import JetDescriptor
from ipcv.misc import read_keypoints, write_keypoints

import argparse
description = '''
Extract jet descriptors from an image using a set of keypoints. E.g. 'python
jet_descriptor.py -i data/dturobot01.png -k data/dturobot01.kp -o
dturobot01.desc'
'''
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-i', '--image_file', type=str, required=True,
                    help='Image file path.')
parser.add_argument('-k', '--keypoint_file', type=str, required=True,
                    help='Keypoints file path.')
parser.add_argument('-o', '--output_file', type=str, required=True,
                    help='Output file path')
args = parser.parse_args()


def run():
    img = sp.misc.imread(args.image_file, flatten=True)/255.
    keypoints = read_keypoints(args.keypoint_file)
    jd = JetDescriptor()
    descs = jd.compute(img, keypoints)
    write_keypoints(args.output_file, keypoints, descs)

if __name__ == '__main__':
    run()
