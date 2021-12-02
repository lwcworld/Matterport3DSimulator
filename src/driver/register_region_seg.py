import MatterSim
import math
import cv2
import numpy as np
import copy
import networkx as nx
from matplotlib import pyplot as plt
from random import sample, seed
import json
import glob
import os
import errno

mapname_list = ['17DRP5sb8fy', '1LXtFkjw3qL', '1pXnuDYAj8r', '29hnd4uzFmX', '2azQ1b91cZZ', '2n8kARJN3HM', '2t7WUuJeko7',
                '5LpN3gDmAk7', '5q7pvUzZiYa', '5ZKStnWn8Zo', '759xd9YjKW5', '7y3sRwLe3Va', '8194nk5LbLH', '82sE5b5pLXE',
                '8WUmhLawc2A', 'aayBHfsNo7d', 'ac26ZMwG7aT', 'ARNzJeq3xxb', 'B6ByNegPMKs', 'b8cTxDM8gDG', 'cV4RVeZvu5T']

test = []
for mapname in mapname_list:
    dir_house_seg = '/root/mount/Matterport3DSimulator/data/v1/scans/' + mapname + '/house_segmentations/panorama_to_region.txt'

    dict_viewpoint_to_region = {}
    house_seg_list = []
    with open(dir_house_seg) as my_file:
        for line in my_file:
            house_seg_list.append(line)
    for house_seg in house_seg_list:
        house_seg_split = house_seg.split(' ')
        dict_viewpoint_to_region[house_seg_split[1]] = house_seg_split[3][0]

    dir_log = '/root/mount/Matterport3DSimulator/datalog_roomnav/' + mapname
    json_dir_list = glob.glob(dir_log + "/*.json")
    for json_dir in json_dir_list:
        with open(json_dir, 'r') as fp:
            dict_viewpoint = json.load(fp)
        viewpoint = dict_viewpoint['viewpoint']

        dict_viewpoint['region'] = dict_viewpoint_to_region[viewpoint]
        test.append(dict_viewpoint_to_region[viewpoint])
        with open(json_dir, 'w') as fp:
            json.dump(dict_viewpoint, fp)

print(np.unique(test))
