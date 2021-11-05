#!/usr/bin/env python3.6
import MatterSim
import time
import math
import cv2
import numpy as np
import copy

WIDTH = 450
HEIGHT = 300
VFOV = math.radians(60)
HFOV = VFOV*WIDTH/HEIGHT
TEXT_COLOR = [230, 40, 40]

print('H / V angle : ', round(HFOV*180/np.pi, 2), '/', round(VFOV*180/np.pi, 2))

cv2.namedWindow('panoramic RGB')

sim = MatterSim.Simulator()
sim.setCameraResolution(WIDTH, HEIGHT)
sim.setCameraVFOV(VFOV)
sim.setDepthEnabled(False) # Turn on depth only after running ./scripts/depth_to_skybox.py (see README.md)
sim.setRestrictedNavigation(False)
sim.initialize()
sim.newRandomEpisode(['1LXtFkjw3qL'])

heading = 0
elevation = 0
location = 0
ANGLEDELTA = 5 * math.pi / 180

print('\nPython Demo')
print('Use arrow keys to move the camera.')
print('Use number keys (not numpad) to move to nearby viewpoints indicated in the RGB view.')

def get_circle_headings(heading_now, ang_step=90):
    ang2rad = np.pi/180.0
    headings = [-heading_now]
    for i in range(0,int(360/ang_step)-1):
        headings.append(1*ang_step*ang2rad)
    return headings

def get_pano_togo(sim, num_view):
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
        for idx, loc in enumerate(navloc_local[1:]):
            fontScale = 3.0 / loc.rel_distance
            x = int(WIDTH / 2 + loc.rel_heading / HFOV * WIDTH)
            y = int(HEIGHT / 2 - loc.rel_elevation / VFOV * HEIGHT)
            # cv2.putText(view_locallist[i_h], str(navloc_l2g[i_h][idx + 1]), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
            #             fontScale, TEXT_COLOR, thickness=3)
            cv2.putText(view_locallist[i_h], str(navloc_l2g[i_h][idx + 1]), (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale, TEXT_COLOR, thickness=3)

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

if __name__ == "__main__":
    while True:
        # move
        sim.makeAction([location], [heading], [elevation])
        location, heading, elevation = 0, 0, 0

        # get panoramic observation
        num_view = 4
        view_list, view_pano, navloc_list, navloc_global, navloc_l2g = get_pano_togo(sim, num_view)


        state = sim.getState()[0]
        # print(state.heading)
        locations = state.navigableLocations
        cv2.imshow('panoramic RGB', view_pano)
        k = cv2.waitKey(1)
        if k == -1:
            continue
        else:
            k = (k & 255)
        if k == ord('q'):
            break
        elif ord('1') <= k <= ord('9'):
            location = k - ord('0')
            if location >= len(locations):
                location = 0
        elif k == 81 or k == ord('a'):
            heading = -ANGLEDELTA
        elif k == 82 or k == ord('w'):
            elevation = ANGLEDELTA
        elif k == 83 or k == ord('d'):
            heading = ANGLEDELTA
        elif k == 84 or k == ord('s'):
            elevation = -ANGLEDELTA