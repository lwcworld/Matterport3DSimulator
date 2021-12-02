import json
import numpy as np
import networkx as nx
import cv2
import os

def create_dir(path):
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory is created!")

def get_edge(connectivity_list, way='visible'):
    N = len(connectivity_list)
    A = np.zeros((N, N))
    for i_node, connectivity in enumerate(connectivity_list):
        neighbor_node_list = [node for node, isneighbor in enumerate(connectivity[way]) if isneighbor==True]
        for neighbor_node in neighbor_node_list:
            A[i_node, neighbor_node] = 1
    rows, cols = np.where(A == 1)
    edges = zip(rows.tolist(), cols.tolist())
    return edges

def get_nodes(connectivity_list):
    node_list = []
    for i_node, connectivity in enumerate(connectivity_list):
        attribute = {'image_id':connectivity['image_id'], 'pose':connectivity['pose'], 'included':connectivity['included'], 'height':connectivity['height']}
        node = (i_node, attribute)
        node_list.append(node)
    return node_list

def load_mapname_list(dir):
    f = open(dir, 'r')
    mapname_list = []
    while True:
        line = f.readline()
        if not line: break
        line = line.replace('\n', '')
        mapname_list.append(line)
    return mapname_list

if __name__ == "__main__":
    dir_mapname_list = '/root/mount/Matterport3DSimulator/connectivity/scans.txt'
    dir_connectivity = '/root/mount/Matterport3DSimulator/connectivity/'
    dir_panos = '/root/mount/Matterport3DSimulator/v1/tasks/region_classification/data/mp_sb/'
    dir_save = '/root/mount/Matterport3DSimulator/dataset_topoclassifier/'

    mapname_list = load_mapname_list(dir_mapname_list)

    for mapname in mapname_list:
        # graph construction
        with open(dir_connectivity + mapname + '_connectivity.json', "r") as connectivity:
            connectivity_list = json.load(connectivity)
        nodes = get_nodes(connectivity_list)
        edges = get_edge(connectivity_list, way='visible')
        G = nx.Graph()
        G.add_nodes_from(nodes)
        G.add_edges_from(edges)

        # panorama
        dataset_panorama_dict = {}
        for node in list(G.nodes(data=True)):
            node_id = node[0]
            node_attr = node[1]
            image_id = node_attr['image_id']
            pano = cv2.imread(dir_panos+mapname+'/'+image_id+'.jpg')
            dataset_panorama_dict[node[0]] = pano

        # region info
        dir_house_seg = '/root/mount/Matterport3DSimulator/v1/scans/' + mapname + '/house_segmentations/panorama_to_region.txt'
        dict_viewpoint_to_region = {}
        house_seg_list = []
        with open(dir_house_seg) as my_file:
            for line in my_file:
                house_seg_list.append(line)
        for house_seg in house_seg_list:
            house_seg_split = house_seg.split(' ')
            dict_viewpoint_to_region[house_seg_split[1]] = house_seg_split[3][0]
        for node in list(G.nodes(data=True)):
            node_id = node[0]
            node_attr = node[1]
            info = {"node_id": node_id,
                    "mapname": mapname,
                    "image_id": node_attr['image_id'],
                    "region": dict_viewpoint_to_region[node_attr['image_id']]}
            with open(dir_save + mapname + '/info_' + str(node_id) + '.json', 'w') as fp:
                json.dump(info, fp)

        # save
        dir_save_map = dir_save + mapname + '/'
        create_dir(dir_save_map)
        nx.write_gpickle(G, dir_save_map + 'graph.gpickle')

        for node_id, pano in dataset_panorama_dict.items():
            cv2.imwrite(dir_save_map + 'pano_' + str(node_id) + '.jpg', pano)

