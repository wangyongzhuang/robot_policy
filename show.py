# -*- coding:utf-8 -*-
#Author:Thomas Young ,SJTU ,China

import socket
import sys
import numpy as np
import struct
import os

import numpy as np
import matplotlib.pyplot as plt
from time import time, sleep

from robot_config import *
from robot_environ import *
from robot_client import *

from state_machines.path_planning import PathPlanning#*
from state_machines.norm_attack import NormAttack

import pdb

# init
flag = config()
info_1, info_2, map_img = get_init()
raw_map_img, bars = create_raw_map_img()
#pdb.set_trace()
# info_tmp
info_2 = [[270,280,100,0, 2000], [270,330,100,0, 2000]]
norm_attack_1 = NormAttack(0, info_1, info_2)
norm_attack_2 = NormAttack(1, info_1, info_2)

def show(way1, way2, self_info, enemy_info):
    img = raw_map_img.copy()
    #img[self_info[0][0]-25:self_info[0][0]+25, self_info[0][1]-25:self_info[0][1]+25] = np.array([0,255,0], np.uint8)
    img[self_info[1][0]-25:self_info[1][0]+25, self_info[1][1]-25:self_info[1][1]+25] = np.array([0,0,255], np.uint8)
    img[enemy_info[0][0]-25:enemy_info[0][0]+25, enemy_info[0][1]-25:enemy_info[0][1]+25] = np.array([255,0,0], np.uint8)
    img[enemy_info[1][0]-25:enemy_info[1][0]+25, enemy_info[1][1]-25:enemy_info[1][1]+25] = np.array([255,0,0], np.uint8)
    #for i,w in enumerate(way1[1:-1]):
        #img[w[0]-5:w[0]+5, w[1]-5:w[1]+5] = np.array([0,120,0], np.uint8)
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
    # enemy
    #act_2_p[1, 5] = act_2_p[1, 16] = 0
    act_2_p[0, 5] = act_2_p[0, 16] = 1
    act_2_p[1, 5] = act_2_p[1, 16] = 1
    act_2_p[0, -1] = act_2_p[1, -1] = 0
    #print 'time ',time()-t
    #print 'info', info_1[0][:2], info_1[1][:2]
    if show_info:
        show(norm_attack_1.path_planner.way, norm_attack_2.path_planner.way, info_1, info_2)
    info_1, info_2, r1, r2, map_img_new = environ(flag, info_1, info_2, act_1_p, act_2_p, map_img)

    return info_1, info_2, map_img_new


def draw(cur_pos,dst_pos=None, enemy_pos=None,way=None,img=None):
    img = img.copy()
    img[cur_pos[0]-25:cur_pos[0]+25, cur_pos[1]-25:cur_pos[1]+25] = np.array([0,255,0], np.uint8)
    if enemy_pos is not None:
        img[enemy_pos[0]-25:enemy_pos[0]+25, enemy_pos[1]-25:enemy_pos[1]+25] = np.array([255,0,0], np.uint8)
    if dst_pos is not None:
        img[dst_pos[0]-10:dst_pos[0]+10, dst_pos[1]-10:dst_pos[1]+10] = np.array([0,0,255], np.uint8)
    for i,w in enumerate(way):
        img[w[0]-5:w[0]+5, w[1]-5:w[1]+5] = np.array([0,120,0], np.uint8)
    img[0:300:5,250] = 0
    img[300,250:500:5] = 0
    img = img.transpose((1, 0, 2))
    plt.imshow(img)
    plt.show()

def re(a,n):
    return [a.tolist()+[n],a.tolist()+[n]]

if __name__ == '__main__':
    screen = pygame.display.set_mode((1200, 500), 0, 32)
    raw_map_img, bars = create_raw_map_img()

    enemy_pos = np.array([270, 280])
    cur_pos = np.array([30, 430])

    dst_pos = np.array([120, 280])
    dst_pos1 = np.array([250, 470])
    pp = PathPlanning(cur_pos, dst_pos)
    pp1 = PathPlanning(dst_pos, dst_pos1)
    dp = pp1.run(dst_pos, 14, raw_map_img, strategy='a-star')

    tmp_pos = np.array([170,400])
    pp2 = PathPlanning(tmp_pos, enemy_pos)
    dp = pp2.run(tmp_pos, 14, raw_map_img, strategy='a-star')
    
    pp30_pos = np.array([210,380])
    pp31_pos = np.array([120,280])
    pp3 = PathPlanning(pp30_pos, pp31_pos)
    dp = pp2.run(pp30_pos, 14, raw_map_img, strategy='a-star')

    step = 0
    blood = 100
    while np.linalg.norm(abs(cur_pos - dst_pos), ord=np.inf) > 14:
        step += 1
        t = time()
        dp = pp.run(cur_pos, 14, raw_map_img, strategy='a-star')
        cur_pos += dp
        print('step {} cur pos {} to {}, time {:.2f}'.format(step, cur_pos, dst_pos, time() - t))
        #draw(cur_pos,dst_pos=dst_pos,enemy_pos=enemy_pos.tolist(),way=pp.way,img=raw_map_img)
        draw_info_z(screen, bars, re(cur_pos,100), re(enemy_pos,blood), raw_map_img, dst_pos, pp.way)
        sleep(0.01)

    cur_pos = dst_pos
    step = 0
    #while np.linalg.norm(abs(cur_pos - dst_pos1), ord=np.inf) > 14:
    while step <13:
        step += 1
        t = time()
        dp = pp1.run(cur_pos, 14, raw_map_img, strategy='a-star')
        cur_pos += dp
        #draw(cur_pos,dst_pos=dst_pos1,enemy_pos=enemy_pos.tolist(),way=pp1.way,img=raw_map_img)
        draw_info_z(screen, bars, re(cur_pos,100), re(enemy_pos,blood), raw_map_img, dst_pos1, pp1.way)
        sleep(0.01)
        print('step {} cur pos {} to {}, time {:.2f}'.format(step, cur_pos, dst_pos1, time() - t))

    step = 0
    cur_pos = tmp_pos
    while step<10:
        step += 1
        dp = pp2.run(cur_pos, 14, raw_map_img, strategy='a-star')
        if step <5:
            cur_pos += dp
        else:
            blood -= 1
        #draw(cur_pos,dst_pos=enemy_pos,enemy_pos=enemy_pos.tolist(),way=pp2.way,img=raw_map_img)
        draw_info_z(screen, bars, re(cur_pos,100), re(enemy_pos,blood), raw_map_img, enemy_pos, pp2.way)
        sleep(0.01)
        print('step {} cur pos {} , time {:.2f}'.format(step, cur_pos, time() - t))

    step = 0
    cur_pos = pp30_pos
    while not (cur_pos==pp31_pos).all():
        step += 1
        t = time()
        dp = pp3.run(cur_pos, 14, raw_map_img, strategy='a-star')
        cur_pos += dp
        #draw(cur_pos,dst_pos=pp31_pos,enemy_pos=enemy_pos.tolist(),way=pp3.way,img=raw_map_img)
        draw_info_z(screen, bars, re(cur_pos,100), re(enemy_pos,blood), raw_map_img, pp31_pos, pp3.way)
        sleep(0.01)
        print('step {} cur pos {}, time {:.2f}'.format(step, cur_pos, time() - t))

    print('STOP!')
    os._exit(0)

    '''
    for global_step in range(flag.steps):
        #get_info()
        if global_step<15:
            info_2 = [[80,280,100,0, 2000], [80,280,100,0, 2000]]
        elif global_step<30:
            info_2 = [[250,470,100,0, 2000], [250,470,100,0, 2000]]
        else:
            info_2 = [[270,280,100,0, 2000], [270,330,100,0, 2000]]
        info_1, info_2, map_img = step(flag, info_1, info_2, map_img, show_info=True)
        print 'step ',global_step
        #sendData(conn, info_1)
        #sleep(0.01)
        #pdb.set_trace()
        #draw_info(screen, bars, info_1, info_2, raw_map_img)
    '''

