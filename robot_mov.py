#import tensorflow as tf
import numpy as np
import random
import pygame
import pdb
scale = 0.1
robot_size = [int(500*scale), int(500*scale)]
act_dict = [-12, -8, -4, -2, -1, 0, 1, 2, 4, 8, 12]
d_size = [_/2 for _ in robot_size]

def _sign(x):
    if x>=0:
        return 1
    return -1

def create_raw_map_img():
    img = 255 * np.ones([int(8000*scale), int(5000*scale), 3], dtype=np.float32)
    bars = []
    # bar_1
    img[int(scale*1200):int(scale*2000), int(scale*1000):int(scale*1300), :] = 0
    img[int(scale*6000):int(scale*6800), int(scale*3700):int(scale*4000), :] = 0
    bars.append([int(scale*1200),int(scale*1000), int(scale*(2000-1200)), int(scale*(1300-1000))])
    bars.append([int(scale*6000),int(scale*3700), int(scale*(2000-1200)), int(scale*(1300-1000))])

    # bar_2
    img[int(scale*0):int(scale*800), int(scale*2500):int(scale*2800), :] = 0
    img[int(scale*7200):int(scale*8000), int(scale*2200):int(scale*2500), :] = 0
    bars.append([int(scale*0),int(scale*2500), int(scale*(800-0)), int(scale*(2800-2500))])
    bars.append([int(scale*7200),int(scale*2200), int(scale*(800-0)), int(scale*(2800-2500))])

    # bar_3
    img[int(scale*1800):int(scale*2100), int(scale*2300):int(scale*3500), :] = 0
    img[int(scale*5900):int(scale*6200), int(scale*1500):int(scale*2700), :] = 0
    bars.append([int(scale*1800),int(scale*2300), int(scale*(2100-1800)), int(scale*(3500-2300))])
    bars.append([int(scale*5900),int(scale*1500), int(scale*(2100-1800)), int(scale*(3500-2300))])

    # bar_4
    img[int(scale*3100):int(scale*3400), int(scale*3000):int(scale*5000), :] = 0
    img[int(scale*4600):int(scale*4900), int(scale*0):int(scale*2000), :] = 0
    bars.append([int(scale*3100),int(scale*3000), int(scale*(3400-3100)), int(scale*(5000-3000))])
    bars.append([int(scale*4600),int(scale*0), int(scale*(3400-3100)), int(scale*(5000-3000))])

    return img, bars

def draw_pos_tool(pos, color, map_img):
    map_img[int(pos[0]-d_size[0]):int(pos[0]+d_size[0]), int(pos[1]-d_size[1]):int(pos[1]+d_size[1])] = color
    return map_img

def draw_pos(pos_1, pos_2, raw_map_img):
    map_img = raw_map_img.copy()
    # red
    map_img = draw_pos_tool(pos_1[0][:2], np.array([255, 0, 0],dtype=np.float32), map_img)
    map_img = draw_pos_tool(pos_1[1][2:], np.array([255, 0, 0],dtype=np.float32), map_img)
    # blue
    map_img = draw_pos_tool(pos_2[0][:2], np.array([0, 255, 0],dtype=np.float32), map_img)
    map_img = draw_pos_tool(pos_2[1][2:], np.array([0, 255, 0],dtype=np.float32), map_img)
    return map_img

def _move_tool(pos_pre, dir, map_img):
    flag = False
    pos_new = [pos_pre[0]+dir[0], pos_pre[1]+dir[1]]
    corners = [[pos_new[0]+d_size[0], pos_new[1]+d_size[1]], [pos_new[0]+d_size[0], pos_new[1]-d_size[1]], [pos_new[0]-d_size[0], pos_new[1]+d_size[1]], [pos_new[0]-d_size[0], pos_new[1]-d_size[1]]]
    for corner in corners:
        if sum(map_img[int(corner[0]), int(corner[1])])<765:
            return flag
    return True

def move(pos_pre, dir, map_img):
    # out
    while (pos_pre[0]+_sign(dir[0])*(abs(dir[0])+d_size[0])) < 0 or (pos_pre[0]+_sign(dir[0])*(abs(dir[0])+d_size[0]))>=int(scale*8000):
        if dir[0]==0:
            break
        dir[0] = _sign(dir[0]) * (abs(dir[0]) -1)
    while (pos_pre[1]+_sign(dir[1])*(abs(dir[1])+d_size[1])) < 0 or (pos_pre[1]+_sign(dir[1])*(abs(dir[1])+d_size[1]))>=int(scale*5000):
        if dir[1]==0:
            break
        dir[1] = _sign(dir[1]) * (abs(dir[1]) -1)

    # bar
    while not _move_tool(pos_pre, [dir[0], 0], map_img):
        if dir[0]==0:
            break
        dir[0] = _sign(dir[0]) * (abs(dir[0]) -1)
    while not _move_tool(pos_pre, [0, dir[1]], map_img):
        if dir[1]==0:
            break
        dir[1] = _sign(dir[1]) * (abs(dir[1]) -1)
    return dir

def get_action(act):
    return [act_dict[act[0]], act_dict[act[1]]]


def get_init():
    info_1 = [[30, 400, 100, 0],  [30, 470, 100, 0]]
    info_2 = [[770, 30, 100,0 ], [770, 60, 100, 0]]
    #info_1 = [[30, 30, 100, 0],  [30, 470, 100, 0]]
    #info_2 = [[770, 30, 100,0 ], [770, 470, 100, 0]]
    raw_map_img, _ = create_raw_map_img()

    map_img_new = draw_pos(info_1, info_2, raw_map_img)
    return info_1, info_2, map_img_new

def environ_z(info_1, info_2, act_1_p, act_2_p, raw_map_img):
    # get action
    act_1 = [get_action(act_1[0]), get_action(act_1[1])]
    act_2 = [get_action(act_2[0]), get_action(act_2[1])]

    # move:[dx,dy]
    dir_tmp_1 = [move(info_1[0][:2], act_1[0], map_img), move(info_1[1][:2], act_1[1], map_img)]
    dir_tmp_2 = [move(info_2[0][:2], act_2[0], map_img), move(info_2[1][:2], act_2[1], map_img)]

    # new info
    info_1_new = info_1
    info_2_new = info_2
    info_1_new[0][0] += dir_tmp_1[0][0]
    info_1_new[0][1] += dir_tmp_1[0][1]
    info_1_new[1][0] += dir_tmp_1[1][0]
    info_1_new[1][1] += dir_tmp_1[1][1]

    map_img_new = draw_pos(info_1_new, info_2_new, raw_map_img)

    return info_1_new, info_2_new, map_img_new


if __name__ =='__main__':
    environ_z(info_1, info_2, act_1_p, act_2_p, raw_map_img)