import tensorflow as tf
import numpy as np
import random
import pygame
import pdb
from robot_environ import *
import random 

def agent(flag, info_1, info_2, act_1_p, act_2_p, raw_map_img, policy='RANDOM'):
    # info[2,4]: [x,y,blood,buff]
    # act[2,15]:  [3,2,1,0,-1,-2,-3,3,2,1,0,-1,-2,-3,shoot]

    print 'info:'
    print info_1
    print info_2

    # random
    random_mask = []
    for i in range(8):
        tmp = np.arange(flag.mov_num)
        np.random.shuffle(tmp)
        random_mask.append(tmp)
    random_mask = np.array(random_mask)
    print random_mask

    # sample
    info_1_news = []
    info_2_news = []
    info_1_best = info_1
    info_2_best = info_2
    r_1_best    = 0.
    r_2_best    = 0.
    reward_mask_1 = np.zeros([2, 2*flag.mov_num+1])
    reward_mask_2 = np.zeros([2, 2*flag.mov_num+1])
    for i in range(flag.mov_num):
        act_1 = [random_mask[:2, i].tolist() + [1], random_mask[2:4, i].tolist() + [1]]
        act_2 = [random_mask[4:6, i].tolist() + [1], random_mask[6:, i].tolist() + [1]]

        info_1_new, info_2_new, reward_1, reward_2 = environ(flag, info_1, info_2, act_1, act_2, raw_map_img, policy=policy)

        info_1_news.append(info_1_new)
        info_2_news.append(info_2_new)
        reward_mask_1 = np.maximum(reward_mask_1, reward_1)
        reward_mask_2 = np.maximum(reward_mask_2, reward_2)
        if np.max(reward_1[0]) + np.max(reward_1[1]) > r_1_best:
            r_1_best = np.max(reward_1[0]) + np.max(reward_1[1])
            info_1_best = info_1_new
        if np.max(reward_2[0]) + np.max(reward_2[1]) > r_2_best:
            r_2_best = np.max(reward_2[0]) + np.max(reward_2[1])
            info_2_best = info_2_new

    '''
    act_1 = []
    act_1.append([np.argmax(act_1_p[0,:flag.mov_num]), np.argmax(act_1_p[0,flag.mov_num:2*flag.mov_num]), int(act_1_p[0,-1]>0.5)])
    act_1.append([np.argmax(act_1_p[1,:flag.mov_num]), np.argmax(act_1_p[1,flag.mov_num:2*flag.mov_num]), int(act_1_p[1,-1]>0.5)])

    act_2 = []
    act_2.append([np.argmax(act_2_p[0,:flag.mov_num]), np.argmax(act_2_p[0,flag.mov_num:2*flag.mov_num]), int(act_2_p[0,-1]>0.5)])
    act_2.append([np.argmax(act_2_p[1,:flag.mov_num]), np.argmax(act_2_p[1,flag.mov_num:2*flag.mov_num]), int(act_2_p[1,-1]>0.5)])


    # environ
    info_1_new, info_2_new, reward_1, reward_2, map_img_new = environ(flag, info_1, info_2, act_1, act_2, raw_map_img, policy=policy)
    '''
    if policy == 'MAX':
        info_1_new, info_2_new = info_1_best, info_2_best
    elif policy == 'RANDOM':
        info_1_new = random.choice(info_1_news)
        info_2_new = random.choice(info_2_news)
    #pdb.set_trace()
    map_img_new = draw_pos(info_1_new, info_2_new, raw_map_img)
    
    return info_1_new, info_2_new, reward_mask_1, reward_mask_2, map_img_new