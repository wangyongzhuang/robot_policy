# -*- coding:utf-8 -*-
#Author:Thomas Young ,SJTU ,China

import socket
import sys
import numpy as np
import struct

import numpy as np
import matplotlib.pyplot as plt
from time import time, sleep

from robot_config import *
from robot_environ import *
from robot_client import *

from state_machines.norm_attack import NormAttack

import pdb
act_dict = [-12, -8, -4, -2, -1, 0, 1, 2, 4, 8, 12]

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
    # enemy
    #act_2_p[1, 5] = act_2_p[1, 16] = 0
    act_2_p[0, 3] = act_2_p[0, 16] = 1
    act_2_p[1, 5] = act_2_p[1, 18] = 1
    act_2_p[0, -1] = act_2_p[1, -1] = 0
    #print 'time ',time()-t
    #print 'info', info_1[0][:2], info_1[1][:2]
    if show_info:
        show(norm_attack_1.path_planner.way, norm_attack_2.path_planner.way, info_1, info_2)
    info_1, info_2, r1, r2, map_img_new = environ(flag, info_1, info_2, act_1_p, act_2_p, map_img)

    return info_1, info_2, map_img_new

def get_info(data):
    return info_1, info_2



def sendData(conn, info_1, info_2):
    print 'Receieve data ...'
    data = conn.recv(1024).encode('hex')
    print data

    #info_1 = get_info(data)
    #info_1, info_2, r1, r2, map_img_new = step(flag, info_1, info_2, map_img, show_info=False)

    print 'Send data ...'
    #pdb.set_trace()
    data = '@%03d%03d%03d%03d%%'%(info[0][0], info[0][1], info[1][0], info[1][1])
    #data = '@%03d%03d%03d%03d%%'%(info_1[0][0], info_1[0][1], info_1[1][0], info_1[1][1])
    #struct.pack('B',int('1',16))

    conn.send(data)
    print data
    print 'Send Done.'

def start_tcp_server(ip, port):
    #create socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (ip, port)
    #bind port
    print 'starting listen on ip %s, port %s'%server_address
    sock.bind(server_address)
    #starting listening, allow only one connection
    try:
        sock.listen(1)
    except socket.error, e:
        print "fail to listen on port %s"%e
        sys.exit(1)
    print "waiting for connection"
    conn, addr = sock.accept()

    return conn

if __name__ == '__main__':
    #conn = start_tcp_server('192.168.137.121',10001)
    screen = pygame.display.set_mode((1200, 500), 0, 32)
    raw_map_img, bars = create_raw_map_img()
    for global_step in range(flag.steps):
        #get_info()
        info_1, info_2, map_img = step(flag, info_1, info_2, map_img, show_info=True)
        #sendData(conn, info_1)
        #sleep(0.5)
        #draw_info(screen, bars, info_1, info_2, raw_map_img)