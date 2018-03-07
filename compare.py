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
    target_img_color = cv2.imread(target_img_path)

    # detector_o = cv2.ORB_create()
    # detector_a = cv2.AKAZE_create()
    detector_s = cv2.xfeatures2d.SIFT_create()
    # detector_su = cv2.xfeatures2d.SURF_create()

    # FLANN parameters
    search_params = dict(checks=50)   # or pass empty dictionary
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    FLANN_INDEX_LSH = 5
    # index_params = dict(algorithm = FLANN_INDEX_LSH,
    #                     table_number = 6, # 12
    #                     key_size = 12,     # 20
    #                     multi_probe_level = 1) #2
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # find the keypoints and descriptors in target
    # (target_kp_o, target_des_o) = detector_o.detectAndCompute(target_img, None)
    # (target_kp_a, target_des_a) = detector_a.detectAndCompute(target_img, None)
    # SIFT
    (target_kp_s, target_des_s) = detector_s.detectAndCompute(target_img, None)
    # SURF
    # (target_kp_su, target_des_su) = detector_su.detectAndCompute(target_img, None)

    # get target histogram
    target_hist = cv2.calcHist([target_img_color], [0], None, [256], [0, 256])

    # create BFMatcher objects for orb, akaze and sift
    # bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    # SIFT
    # bf_s = cv2.BFMatcher()

    files = [f for f in os.listdir(img_dir) if re.match(r'.*\.jpg', f)]
    found_in = []
    hist = 0
    for file in files:
        comparing_img_path = os.path.join(img_dir, file)
        good_s = []
        good_su = []
        good_f = []
        try:
            # load comparing images (greyscale for knn and color for histogram)
            comparing_img = cv2.imread(comparing_img_path, cv2.IMREAD_GRAYSCALE)
            comparing_img_color = cv2.imread(comparing_img_path)

            # find the keypoints and descriptors with ORB and AKAZE
            # (comparing_kp_o, comparing_des_o) = detector_o.detectAndCompute(comparing_img, None)
            # (comparing_kp_a, comparing_des_a) = detector_a.detectAndCompute(comparing_img, None)
            # SIFT
            (comparing_kp_s, comparing_des_s) = detector_s.detectAndCompute(comparing_img, None)
            # SURF
            # (comparing_kp_su, comparing_des_su) = detector_su.detectAndCompute(comparing_img, None)

            # Match descriptors with ORB or AKAZE
            # matches_o = bf.match(target_des_o, comparing_des_o)
            # matches_a = bf.match(target_des_a, comparing_des_a)

            # SIFT with BF
            # matches_s = bf_s.knnMatch(target_des_s, comparing_des_s, k=2)
            # SIFT and surf with flann
            matches_f = flann.knnMatch(target_des_s, comparing_des_s,k=2)
            # SURF
            # matches_su = flann.knnMatch(target_des_su, comparing_des_su, k=2)

            # get comparing histogram
            comparing_hist = cv2.calcHist([comparing_img_color], [0], None, [256], [0, 256])

            # # Apply ratio test (SIFT with BF)
            # for m,n in matches_s:
            #     if m.distance < 0.75*n.distance:
            #         good_s.append([m])

            # # Apply ratio test (SURF)
            # for m,n in matches_su:
            #     if m.distance < 0.75*n.distance:
            #         good_su.append([m])

            # ratio test as per Lowe's paper (SIFT with FLANN)
            for i,(m,n) in enumerate(matches_f):
                if m.distance < 0.7*n.distance:
                    good_f.append([m])
                    # matchesMask[i]=[1,0]

            # check distance orb
            # dist_o = [m.distance for m in matches_o]
            # ret_o = round(sum(dist_o) / len(dist_o) if len(dist_o) else 0, 2)
            # check distance akaze
            # dist_a = [m.distance for m in matches_a]
            # ret_a = round(sum(dist_a) / len(dist_a) if len(dist_a) else 0, 2)

            # compare histograms
            hist = round(cv2.compareHist(target_hist, comparing_hist, 0), 2)
        except cv2.error:
            ret_o = 100000
            ret_a = 100000
        # siftres  = len(good_s)
        # surfres  = len(good_su)
        flannres = len(good_f)
        # print(file, " ", "\tORB:", ret_o, "\tAKAE:", ret_a, "\tSIFT:", siftres, "\tSURF:", surfres, "\tFLANN:", flannres, "\tHIST:", hist)
        print(file, " ", "\tSIFT:", flannres, "\tHIST:", hist)
        # check thresholds
        if flannres > 70 or (hist >= 0.7):
            found_in.append(file)
    # print result
    print(os.path.basename(target_img_path), "found in:", found_in)

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
