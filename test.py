import numpy as np
import matplotlib.pyplot as plt
from time import time, sleep
import serial

from robot_config import *
from robot_environ import *
#z# from robot_client import *

from state_machines.norm_attack import NormAttack

import pdb
act_dict = [-12, -8, -4, -2, -1, 0, 1, 2, 4, 8, 12]
# get_init()
#info_1 = [[30, 30, 100, 0],  [30, 470, 100, 0]]
#info_2 = [[770, 30, 100,0 ], [770, 470, 100, 0]]

# init
flag = config()
info_1, info_2, map_img = get_init()
info_2[0:1][0:1] = np.zeros((2, 2))
raw_map_img, bars = create_raw_map_img()
#pdb.set_trace()
#z# tmp
info_2 = [[270,280,100,0], [270,310,100,0]]
norm_attack_1 = NormAttack(0, info_1, info_2)
norm_attack_2 = NormAttack(1, info_1, info_2)

def show(way1, way2, self_info, enemy_info):
    img = raw_map_img.copy()
    img[self_info[0][0]-25:self_info[0][0]+25, self_info[0][1]-25:self_info[0][1]+25] = np.array([0,255,0], np.uint8)
    img[self_info[1][0]-25:self_info[1][0]+25, self_info[1][1]-25:self_info[1][1]+25] = np.array([0,0,255], np.uint8)
    img[enemy_info[0][0]-25:enemy_info[0][0]+25, enemy_info[0][1]-25:enemy_info[0][1]+25] = np.array([255,0,0], np.uint8)
    img[enemy_info[1][0]-25:enemy_info[1][0]+25, enemy_info[1][1]-25:enemy_info[1][1]+25] = np.array([255,0,0], np.uint8)
    for i,w in enumerate(way1[1:-1]):
        img[w[0]-5:w[0]+5, w[1]-5:w[1]+5] = np.array([0,120,0], np.uint8)
    for i,w in enumerate(way2[1:-1]):
        img[w[0]-5:w[0]+5, w[1]-5:w[1]+5] = np.array([0,0,120], np.uint8)
    img = img.transpose((1, 0, 2))
    plt.imshow(img)
    plt.show()

# step test
def step(flag, info_1, info_2=None, map_img=None, show_info=False):
    tmp_1,tmp_2,tmp_map = get_init()
    if info_2 is None:
        info_2 = tmp_2
    if map_img is None:
        map_img = tmp_map
    t = time()
    act_1_p = np.array([norm_attack_1.run(info_1, info_2), norm_attack_2.run(info_1, info_2)])

    act_2_p = np.zeros((2, flag.mov_num * 2 + 1), np.int64)
    act_2_p[:, 0] = act_2_p[:, -1] = act_2_p[0, -2] = act_2_p[1, flag.mov_num] = 1
    print 'time ',time()-t
    if show_info:
        show(norm_attack_1.path_planner.way, norm_attack_2.path_planner.way, info_1, info_2)
    info_1, info_2, r1, r2, map_img_new = environ(flag, info_1, info_2, act_1_p, act_2_p, map_img)

    return info_1, info_2, map_img_new

def get_info():
    return info_1, info_2


if __name__ == '__main__':

    for global_step in range(flag.steps):
        #get_info()
        info_1, info_2, map_img = step(flag, info_1, info_2, map_img, show_info=True)

