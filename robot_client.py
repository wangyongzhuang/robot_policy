from robot_environ import create_raw_map_img
import pygame
from robot_config import *
import numpy as np
raw_map_img, bars = create_raw_map_img()
scale = 0.1
robot_size = [int(500*scale), int(500*scale)]
pygame.init()
screen = pygame.display.set_mode((800, 500), 0, 32)
flag = config()

BLACK = [0, 0, 0]
WHITE = [255, 255, 255]
RED   = [255, 0, 0]
GREEN = [0, 255, 0]
BLUE  = [0, 0, 255]
# Yelly modification: with the global variable 'bars' initializaed by create_raw_map_img(), 
# no need to pass bars as parameter to functions in this file.

def draw_init(screen):
    screen.fill([255,255,255])

    for i in range(len(bars)):
        pygame.draw.rect(screen, [0,0,0], bars[i], 0)

    pygame.display.update()

def draw_robot(screen, pos, color=[255,0,0]):
    rect = [pos[0]-robot_size[0]/2, pos[1]-robot_size[1]/2, robot_size[0], robot_size[1]]
    pygame.draw.rect(screen, color, rect, 0)

# 被打中，中心区域变黑
def draw_robot_shot(screen, pos):
    rect = [pos[0] - robot_size[0]/10, pos[1] - robot_size[1]/10, robot_size[0]/5, robot_size[1]/5]
    pygame.draw.rect(screen, BLACK, rect)

def draw_blood(screen, pos, blood):
    blood_pos = pos
    blood_pos[0] -= robot_size[0]/2
    blood_pos[1] -= (robot_size[1]/2 + 10)
    blood_lenth = blood/100*robot_size[0]
    rect = [blood_pos[0], blood_pos[1], blood_lenth, 8]
    pygame.draw.rect(screen, [255, 0, 0], rect)

    font = pygame.font.SysFont("simsunnsimsun", 12)
    text_blood = font.render("%d"%blood, False, RED)
    screen.blit(text_blood, (blood_pos[0] + blood_lenth + 2, blood_pos[1]))

def shoot(pos, target, map_img):
    dx   = int(robot_size[0]/10)
    dy   = int(robot_size[0]/10)
    targets = [[target[0]+dx, target[1]+dy], [target[0]+dx, target[1]-dy], [target[0]-dx, target[1]+dy], [target[0]-dx, target[1]-dy]]
    for tar in targets:
        for tmp_x in range(pos[0], tar[0]):
            tmp_y = pos[1] + int((tar[1]-pos[1])/(tar[0]-pos[0])) * (tmp_x-pos[0])
            if sum(map_img[tmp_x, tmp_y]) < 2 * 255:
                return False
    return True

def draw_shoot(screen, state_1, state_2):
    shoot_flag = []
    # 我方1能打中敌方1
    shoot_flag.append(shoot([state_1[0]['x'], state_1[0]['y']], [state_2[0]['x'], state_2[0]['y']], raw_map_img))
    # 我方2能打中敌方1
    shoot_flag.append(shoot([state_1[1]['x'], state_1[1]['y']], [state_2[0]['x'], state_2[0]['y']], raw_map_img))
    # 敌方2能打中我方1
    shoot_flag.append(shoot([state_2[1]['x'], state_2[1]['y']], [state_1[0]['x'], state_1[0]['y']], raw_map_img))

    if state_1[0]['shoot']:
        mid_point = [(2*state_1[0]['x'] + state_2[0]['x'])/3, (2*state_1[0]['y'] + state_2[0]['y'])/3]
        # 我方1击中敌方,用黑线连接，被击中者中间区域用黑线表示
        if state_1[0]['hit']:
            if shoot_flag[0]:  # 击中敌方1
                pygame.draw.line(screen, BLACK, [state_1[0]['x'], state_1[0]['y']], [state_2[0]['x'], state_2[0]['y']], 2)
                draw_robot_shot(screen, [state_2[0]['x'], state_2[0]['y']])
            else:              # 击中敌方2
                pygame.draw.line(screen, BLACK, [state_1[0]['x'], state_1[0]['y']], [state_2[1]['x'], state_2[1]['y']], 2)
                draw_robot_shot(screen, [state_2[1]['x'], state_2[1]['y']])
        else:  # 射出子弹，没有打中，用一定长度的蓝线表示
            pygame.draw.line(screen, BLUE, [state_1[0]['x'], state_1[0]['y']], mid_point, 4)
    if state_1[1]['shoot']:
        mid_point = [(2*state_1[1]['x'] + state_2[0]['x'])/3, (2*state_1[1]['y'] + state_2[0]['y'])/3]
        # 我方2击中敌方
        if state_1[1]['hit']:
            if shoot_flag[1]:
                pygame.draw.line(screen, BLACK, [state_1[1]['x'], state_1[1]['y']], [state_2[0]['x'], state_2[0]['y']], 2)
                draw_robot_shot(screen, [state_2[0]['x'], state_2[0]['y']])
            else:
                pygame.draw.line(screen, BLACK, [state_1[1]['x'], state_1[1]['y']], [state_2[1]['x'], state_2[1]['y']], 2)
                draw_robot_shot(screen, [state_2[1]['x'], state_2[1]['y']])
        else:
            pygame.draw.line(screen, BLUE, [state_1[1]['x'], state_1[1]['y']], mid_point, 4)
    if state_2[0]['shoot']:
        mid_point = [(2*state_2[0]['x'] + state_1[0]['x']) / 3, (2*state_2[0]['y'] + state_1[0]['y']) / 3]
        # 敌方1击中我方
        if state_2[0]['hit']:
            if shoot_flag[0]:
                pygame.draw.line(screen, BLACK, [state_2[0]['x'], state_2[0]['y']], [state_1[0]['x'], state_1[0]['y']], 2)
                draw_robot_shot(screen, [state_1[0]['x'], state_1[0]['y']])
            else:
                pygame.draw.line(screen, BLACK, [state_2[0]['x'], state_2[0]['y']], [state_1[1]['x'], state_1[1]['y']], 2)
                draw_robot_shot(screen, [state_1[1]['x'], state_1[1]['y']])
        else:
            pygame.draw.line(screen, BLUE, [state_2[0]['x'], state_2[0]['y']], mid_point, 4)
    if state_2[1]['shoot']:
        mid_point = [(2*state_2[1]['x'] + state_1[0]['x']) / 3, (2*state_2[1]['y'] + state_1[0]['y']) / 3]
        # 敌方2击中我方
        if state_2[1]['hit']:
            if shoot_flag[2]:
                pygame.draw.line(screen, BLACK, [state_2[1]['x'], state_2[1]['y']], [state_1[0]['x'], state_1[0]['y']], 2)
                draw_robot_shot(screen, [state_1[0]['x'], state_1[0]['y']])
            else:
                pygame.draw.line(screen, BLACK, [state_2[1]['x'], state_2[1]['y']], [state_1[1]['x'], state_1[1]['y']], 2)
                draw_robot_shot(screen, [state_1[1]['x'], state_1[1]['y']])
        else:
            pygame.draw.line(screen, BLUE, [state_2[1]['x'], state_2[1]['y']], mid_point, 4)

def draw_act(screen, pos, state):
    dx = state['dx']
    dy = state['dy']

    pos_new = [pos[0] + dx, pos[1] + dy]
    pygame.draw.line(screen, (0, 0, 0), pos, pos_new, 3)
    font = pygame.font.SysFont("simsunnsimsun", 16)
    text_act = font.render("(%d, %d)" % (dx, dy), False, BLACK)
    screen.blit(text_act, pos_new)

def draw_state(screen, bars, state_1, state_2, info_1, info_2):
    draw_init(screen, bars)

    draw_robot(screen, [state_1[0]['x'], state_1[0]['y']], [255,0,0])
    draw_robot(screen, [state_1[1]['x'], state_1[1]['y']], [255,0,0])
    draw_robot(screen, [state_2[0]['x'], state_2[0]['y']], [0,255,0])
    draw_robot(screen, [state_2[1]['x'], state_2[1]['y']], [0,255,0])

    draw_blood(screen, [state_1[0]['x'], state_1[0]['y']], info_1[0][2])
    draw_blood(screen, [state_1[1]['x'], state_1[1]['y']], info_1[1][2])
    draw_blood(screen, [state_2[0]['x'], state_2[0]['y']], info_2[0][2])
    draw_blood(screen, [state_2[1]['x'], state_2[1]['y']], info_2[1][2])

    draw_act(screen, [state_1[0]['x'], state_1[0]['y']], state_1[0])
    draw_act(screen, [state_1[1]['x'], state_1[1]['y']], state_1[1])
    draw_act(screen, [state_2[0]['x'], state_2[0]['y']], state_2[0])
    draw_act(screen, [state_2[1]['x'], state_2[1]['y']], state_2[1])

    draw_shoot(screen, state_1, state_2)

    pygame.display.update()

def draw_info(screen, bars, info_1, info_2):
    draw_init(screen, bars)

    draw_robot(screen, info_1[0][:2], RED)
    draw_robot(screen, info_1[1][:2], RED)
    draw_robot(screen, info_2[0][:2], GREEN)
    draw_robot(screen, info_2[1][:2], GREEN)

    draw_blood(screen, info_1[0][:2], info_1[0][2])
    draw_blood(screen, info_1[1][:2], info_1[1][2])
    draw_blood(screen, info_2[0][:2], info_2[0][2])
    draw_blood(screen, info_2[1][:2], info_2[1][2])

    pygame.display.update()
