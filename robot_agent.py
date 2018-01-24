import tensorflow as tf
import numpy as np
import random
import pygame
import pdb
from robot_environ import *

def agent(flag, info_1, info_2, act_1_p, act_2_p, raw_map_img, policy='MAX'):
    # info[2,4]: [x,y,blood,buff]
    # act[2,15]:  [3,2,1,0,-1,-2,-3,3,2,1,0,-1,-2,-3,shoot]

    print 'info:'
    print info_1
    print info_2

    act_1 = []
    act_1.append([np.argmax(act_1_p[0,:flag.mov_num]), np.argmax(act_1_p[0,flag.mov_num:2*flag.mov_num]), int(act_1_p[0,-1]>0.5)])
    act_1.append([np.argmax(act_1_p[1,:flag.mov_num]), np.argmax(act_1_p[1,flag.mov_num:2*flag.mov_num]), int(act_1_p[1,-1]>0.5)])

    act_2 = []
    act_2.append([np.argmax(act_2_p[0,:flag.mov_num]), np.argmax(act_2_p[0,flag.mov_num:2*flag.mov_num]), int(act_2_p[0,-1]>0.5)])
    act_2.append([np.argmax(act_2_p[1,:flag.mov_num]), np.argmax(act_2_p[1,flag.mov_num:2*flag.mov_num]), int(act_2_p[1,-1]>0.5)])


    # environ
    info_1_new, info_2_new, reward_1, reward_2, map_img_new = environ(flag, info_1, info_2, act_1, act_2, raw_map_img, policy=policy)



    return info_1_new, info_2_new, reward_1, reward_2, map_img_new