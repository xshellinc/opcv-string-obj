#!/usr/bin/env python2

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import argparse
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)
import glob
import logging
import os
import re
import sys
import time
import errno
import numpy as np


# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

# help(cv2.xfeatures2d)

def parse_args():
    parser = argparse.ArgumentParser(description='End-to-end inference')
    parser.add_argument(
        '--output-dir',
        dest='output_dir',
        help='directory for visualization pdfs (default: /tmp/infer_simple)',
        default='output',
        type=str
    )
    parser.add_argument(
        '--image-ext',
        dest='image_ext',
        help='image file name extension (default: jpg)',
        default='jpg',
        type=str
    )
    parser.add_argument(
        'im_or_folder', help='image or folder of images', default=None
    )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def compare_feature(target_img_path, img_dir):
    target_img = cv2.imread(target_img_path, cv2.IMREAD_GRAYSCALE)
    IMG_SIZE = target_img.shape
    # target_img = cv2.resize(target_img, IMG_SIZE)

    # bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    detector_o = cv2.ORB_create()
    detector_s = cv2.xfeatures2d.SIFT_create()
    detector_a = cv2.AKAZE_create()
    
    # find the keypoints and descriptors
    (target_kp_o, target_des_o) = detector_o.detectAndCompute(target_img, None)
    (target_kp_s, target_des_s) = detector_s.detectAndCompute(target_img, None)
    (target_kp_a, target_des_a) = detector_a.detectAndCompute(target_img, None)
    # create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    bf_s = cv2.BFMatcher()

    files = [f for f in os.listdir(img_dir) if re.match(r'.*\.jpg', f)]
    for file in files:
        comparing_img_path = os.path.join(img_dir, file)
        good = []
        try:
            comparing_img = cv2.imread(comparing_img_path, cv2.IMREAD_GRAYSCALE)
            # comparing_img = cv2.resize(comparing_img, IMG_SIZE)
            # find the keypoints and descriptors
            (comparing_kp_o, comparing_des_o) = detector_o.detectAndCompute(comparing_img, None)
            (comparing_kp_s, comparing_des_s) = detector_s.detectAndCompute(comparing_img, None)
            (comparing_kp_a, comparing_des_a) = detector_a.detectAndCompute(comparing_img, None)
            # Match descriptors with ORB or AKAZE
            matches_o = bf.match(target_des_o, comparing_des_o)
            matches_a = bf.match(target_des_a, comparing_des_a)
            matches_s = bf_s.knnMatch(target_des_s, comparing_des_s, k=2)

            # Apply ratio test
            for m,n in matches_s:
                if m.distance < 0.75*n.distance:
                    good.append([m])

            # check distance orb
            dist_o = [m.distance for m in matches_o]
            ret_o = sum(dist_o) / len(dist_o) if len(dist_o) else 0
            # check distance akaze
            dist_a = [m.distance for m in matches_a]
            ret_a = sum(dist_a) / len(dist_a) if len(dist_a) else 0
        except cv2.error:
            ret_o = 100000
            ret_a = 100000

        print(file, ret_o, ret_a, len(good))

def compare_hist(target_img_path, img_dir):
    target_img = cv2.imread(target_img_path)
    IMG_SIZE = target_img.shape
    # target_img = cv2.resize(target_img, IMG_SIZE)
    target_hist = cv2.calcHist([target_img], [0], None, [256], [0, 256])

    files = [f for f in os.listdir('.') if re.match(r'.*\.jpg', f)]
    for file in files:
        comparing_img_path = img_dir + file
        comparing_img = cv2.imread(comparing_img_path)
        comparing_img = cv2.resize(comparing_img, IMG_SIZE)
        comparing_hist = cv2.calcHist([comparing_img], [0], None, [256], [0, 256])

        ret = cv2.compareHist(target_hist, comparing_hist, 0)
        print(file, ret)
    

def main(args):
    logger = logging.getLogger(__name__)
    # enumerate extracted objects
    im_list = glob.iglob(os.path.join(args.output_dir, "car") + '/*.jpg')
    # for each object
    for i, im_name in enumerate(im_list):
        logger.info(im_name)
        # search for that image in target folder
        compare_feature(im_name, args.im_or_folder)

if __name__ == '__main__':
    FORMAT = '%(levelname)s %(filename)s:%(lineno)4d: %(message)s'
    # Manually clear root loggers to prevent any module that may have called
    # logging.basicConfig() from blocking our logging setup
    logging.root.handlers = []
    logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
    args = parse_args()
    main(args)
