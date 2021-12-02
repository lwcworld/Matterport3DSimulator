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

def sim_init(param_sim, mapname, random_initpos=True):
    sim = MatterSim.Simulator()
    sim.setCameraResolution(param_sim['WIDTH'], param_sim['HEIGHT'])
    sim.setCameraVFOV(param_sim['VFOV'])
    sim.setDepthEnabled(False)  # Turn on depth only after running ./scripts/depth_to_skybox.py (see README.md)
    sim.setRestrictedNavigation(False)
    sim.initialize()

    if random_initpos==True:
        dir_viewpoints = "/root/mount/Matterport3DSimulator/data/v1/scans/" + mapname[0] + "/matterport_skybox_images/*"
        viewpoints = [os.path.basename(x) for x in glob.glob(dir_viewpoints)]
        viewpoint = sample(viewpoints,1)[0]
        viewpoint_parse = viewpoint.split("_")[0]
        sim.newEpisode(mapname, [viewpoint_parse], [0], [0])
    else:
        sim.newRandomEpisode(mapname)

    print('map : ', mapname, ' H/V angle : ', round(param_sim['HFOV'] * 180 / np.pi, 2), '/', round(param_sim['VFOV'] * 180 / np.pi, 2))
    return sim

def get_circle_headings(heading_now, ang_step=90):
    ang2rad = np.pi/180.0
    headings = [-heading_now]
    for i in range(0,int(360/ang_step)-1):
        headings.append(1*ang_step*ang2rad)
    return headings

def get_navloc_info(sim, num_view, param_sim):
    state = sim.getState()[0]
    headings = get_circle_headings(state.heading, ang_step=360/num_view)
    view_locallist = []
    navloc_locallist = []
    navloc_global = []
    navloc_l2g = [] # local to global matcher
    ixs_navloc = []
    location, elevation = 0, 0
    for i_h, heading in enumerate(headings):
        sim.makeAction([location], [heading], [elevation])
        state = sim.getState()[0]

        # get viewpoint
        rgb = np.array(state.rgb, copy=False)
        view_locallist.append(copy.deepcopy(rgb))

        # get navigable location (navloc)
        navloc_local = state.navigableLocations
        navloc_locallist.append(navloc_local)

    # get navloc_global
    navloc_local = navloc_locallist[0]
    for _, navloc in enumerate(navloc_local):
        if navloc.ix not in ixs_navloc:
            ixs_navloc.append(navloc.ix)
            navloc_global.append(navloc)

    # get local togo to global togo
    for i, navloc_local in enumerate(navloc_locallist):
        navloc_l2g.append({})
        for i_local, navloc in enumerate(navloc_local):
            i_global = ixs_navloc.index(navloc.ix)
            navloc_l2g[i][i_local] = i_global

    # add togo texts
    state = sim.getState()[0]
    headings = get_circle_headings(state.heading, ang_step=360/num_view)
    for i_h, heading in enumerate(headings):
        sim.makeAction([location], [heading], [elevation])
        navloc_local = state.navigableLocations
        if param_sim['write_text_togo']:
            for idx, loc in enumerate(navloc_local[1:]):
                fontScale = 3.0 / loc.rel_distance
                x = int(param_sim['WIDTH'] / 2 + loc.rel_heading / param_sim['HFOV'] * param_sim['WIDTH'])
                y = int(HEIGHT / 2 - loc.rel_elevation / param_sim['VFOV'] * param_sim['HEIGHT'])
                cv2.putText(view_locallist[i_h], str(navloc_l2g[i_h][idx + 1]), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale, param_sim['TEXT_COLOR'], thickness=3)

    # get panoramic view
    for i, view_local in enumerate(view_locallist):
        if i == 0:
            view_pano = copy.deepcopy(view_local)
        else:
            view_pano = np.hstack((view_pano, copy.deepcopy(view_local)))

    # re-aline view to 0 angle
    state = sim.getState()[0]
    location, heading, elevation = 0, state.heading, 0
    sim.makeAction([location], [-heading], [elevation])

    return view_locallist, view_pano, navloc_locallist, navloc_global, navloc_l2g

def update_nav_graph(G, navloc_global):
    dict_ix_global = nx.get_node_attributes(G, 'ix_global')
    # reset all local indexs
    for i in G.nodes():
        G.node[i]['ix_local']=None

    if nx.number_of_nodes(G)>=1:
        ix_isrobot = [node for node, attr in G.nodes(data=True) if attr['isrobot'] == True][0]
        G.node[ix_isrobot]['color'] = 'b'
        G.node[ix_isrobot]['isrobot'] = False

    for i, navloc in enumerate(navloc_global):
        ix = navloc.ix
        if i == 0:
            ix_o = copy.deepcopy(ix)

        if ix not in list(dict_ix_global.values()):
            id = navloc.viewpointId
            ix = navloc.ix
            x, y, z = navloc.x, navloc.y, navloc.z
            G.add_nodes_from([ix], id=id, ix_global=ix, ix_local=i, pos=(x, y), isrobot=False, isopen=True, color='y', visit=False)
            G.add_edge(ix_o, ix)
        else:
            if G.has_edge(ix_o, ix) == False:
                G.add_edge(ix_o, ix)
            G.node[ix]['ix_local'] = i

    G.node[ix_o]['visit'] = True
    G.node[ix_o]['isrobot'] = True
    G.node[ix_o]['isopen'] = False
    G.node[ix_o]['color'] = 'r'
    return G

def create_dir(path):
    isExist = os.path.exists(path)
    if not isExist:
        # Create a new directory because it does not exist
        os.makedirs(path)
        print("The new directory is created!")

if __name__ == "__main__":
    # declare cv2 (for paranomic view)and plt (for graph) windows
    cv2.namedWindow('panoramic RGB')
    plt.ion();
    fig = plt.figure();
    ax = fig.add_subplot(111);
    fig.canvas.draw()

    WIDTH, HEIGHT, VFOV = 473, 473, math.radians(60)
    param_sim = {'WIDTH': WIDTH, 'HEIGHT': HEIGHT, 'VFOV': VFOV, 'HFOV': VFOV * WIDTH / HEIGHT, 'TEXT_COLOR': [230, 40, 40],
                 'write_text_togo': False, 'randomwalk': True, 'onlyneighbor': False}

    mapname_list = ['17DRP5sb8fy', '1LXtFkjw3qL', '1pXnuDYAj8r', '29hnd4uzFmX', '2azQ1b91cZZ', '2n8kARJN3HM', '2t7WUuJeko7',
                    '5LpN3gDmAk7', '5q7pvUzZiYa', '5ZKStnWn8Zo', '759xd9YjKW5', '7y3sRwLe3Va', '8194nk5LbLH', '82sE5b5pLXE',
                    '8WUmhLawc2A', 'aayBHfsNo7d', 'ac26ZMwG7aT', 'ARNzJeq3xxb', 'B6ByNegPMKs', 'b8cTxDM8gDG', 'cV4RVeZvu5T']

    max_episode = len(mapname_list)
    max_step = 50000

    i_episode = 0
    while i_episode < max_episode:
        # clear figure
        plt.clf()
        ax = fig.add_subplot(111)
        fig.canvas.draw()

        # init sim variable
        isinit = False
        if i_episode > 0:
            del sim
        while not isinit:
            try:
                mapname_sample = [mapname_list[i_episode]]
                sim = sim_init(param_sim, mapname_sample, random_initpos=True)
                isinit = True
            except:
                pass

        # log directory
        dir_log = 'datalog_roomnav_onlyimg/' + mapname_sample[0]
        create_dir(dir_log)

        # load viewpoint to region matching information
        dir_house_seg = '/root/mount/Matterport3DSimulator/data/v1/scans/' + mapname_sample[0] + '/house_segmentations/panorama_to_region.txt'
        dict_viewpoint_to_region = {}
        house_seg_list = []
        with open(dir_house_seg) as my_file:
            for line in my_file:
                house_seg_list.append(line)
        for house_seg in house_seg_list:
            house_seg_split = house_seg.split(' ')
            dict_viewpoint_to_region[house_seg_split[1]] = house_seg_split[3][0]

        # init iteration variables
        i_step = 0
        i_journey = 0
        heading, elevation, location = 0, 0, 0

        # init graph
        to_vis = True
        G = nx.Graph()
        while i_step < max_step or len(ixs_nonvisit)==0:
            # move
            sim.makeAction([location], [heading], [elevation])
            location, heading, elevation = 0, 0, 0

            # get panoramic observation
            num_view = 4
            view_list, view_pano, navloc_list, navloc_global, navloc_l2g = get_navloc_info(sim, num_view, param_sim)

            # update navigation graph
            G = update_nav_graph(G, navloc_global)

            # show panoramic view
            if to_vis==True:
                state = sim.getState()[0]

                # save & show panoramic view
                cv2.imwrite(dir_log+'/pano_'+str(state.location.ix)+'.jpg', view_pano)
                cv2.imshow('panoramic RGB', view_pano)

                i_step = i_step + 1
                to_vis = False

            # move
            k = cv2.waitKey(3)
            if k == ord('q'):
                break

            if i_journey == 0:
                dict_visit = nx.get_node_attributes(G, 'visit')
                ixs_nonvisit = [k for k, v in dict_visit.items() if v == False]
                if len(ixs_nonvisit) == 0:
                    break
                ix_robot = [node for node, attr in G.nodes(data=True) if attr['isrobot'] == True][0]
                ix_goal = sample(ixs_nonvisit, 1)[0]
                ixs_visit = [k for k, v in dict_visit.items() if v == True] + [ix_goal]
                G_sub = G.subgraph(ixs_visit)
                path = nx.shortest_path(G_sub, source=ix_robot, target=ix_goal)
                i_journey = i_journey + 1
            location_global = path[i_journey]
            dict_ix_local = nx.get_node_attributes(G, 'ix_local')
            location = dict_ix_local[location_global]
            i_journey = i_journey + 1
            if i_journey == len(path):
                i_journey = 0
                to_vis = True

        i_episode = i_episode + 1