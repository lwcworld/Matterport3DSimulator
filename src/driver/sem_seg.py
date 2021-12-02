from keras_segmentation.pretrained import pspnet_50_ADE_20K
import cv2
import numpy as np
import glob
import copy

def split(pano, N_split=4):
    img_list = np.hsplit(pano, 4)
    return img_list

def load_mapname_list(dir):
    f = open(dir, 'r')
    mapname_list = []
    while True:
        line = f.readline()
        if not line: break
        line = line.replace('\n', '')
        mapname_list.append(line)
    return mapname_list

pretrained_model = pspnet_50_ADE_20K()
dir_mapname_list = '/root/mount/Matterport3DSimulator/connectivity/scans.txt'
dir_pano = '../../dataset_topoclassifier/'
dir_save = '../../dataset_topoclassifier/'

mapname_list = load_mapname_list(dir_mapname_list)

N_split = 4
for mapname in mapname_list:
    print('===== mapname : '+mapname)
    pano_dir_list = glob.glob(dir_pano+mapname+'/pano_*.jpg')
    for pano_dir in pano_dir_list:
        node = pano_dir.split(".")[-2].split("_")[-1]
        pano = cv2.imread(pano_dir)

        img_list = split(pano, N_split=N_split)
        for i, img in enumerate(img_list):
            # img_resized = cv2.resize(img, dsize=(473, 473), interpolation=cv2.INTER_NEAREST)
            cv2.imwrite('pano_split_'+str(i)+'.jpg', img)
        for i in range(N_split):
            out = pretrained_model.predict_segmentation(inp = 'pano_split_'+str(i)+'.jpg',
                                                        out_fname = 'bridge.jpg')
            semview = cv2.imread('bridge.jpg')
            if i == 0:
                sem_pano = copy.deepcopy(semview)
            else:
                sem_pano = np.hstack((sem_pano, copy.deepcopy(semview)))
        sem_pano_resized = cv2.resize(sem_pano, dsize=(256,64), interpolation=cv2.INTER_NEAREST)

        cv2.imwrite(dir_save + mapname + '/sempano_' + str(node) + '.jpg', sem_pano)
        cv2.imwrite('sempano_' + str(node) + '.jpg', sem_pano)
        cv2.imwrite(dir_save + mapname + '/sempano_resized_' + str(node) + '.jpg', sem_pano_resized)
        print('saved '+'/sempano_' + str(node))

        break
    break