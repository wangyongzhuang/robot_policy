from robot_environ import create_raw_map_img
import pygame
import numpy as np
raw_map_img, bars = create_raw_map_img()
scale = 0.1
robot_size = [int(500*scale), int(500*scale)]
pygame.init()
screen = pygame.display.set_mode((800, 500), 0, 32)


def draw_init(screen, bars):
    screen.fill([255,255,255])

    for i in range(len(bars)):
        pygame.draw.rect(screen, [0,0,0], bars[i], 0)

    pygame.display.update()

def draw_robot(screen, pos, color=[255,0,0]):
    rect = [pos[0]-robot_size[0]/2, pos[1]-robot_size[1]/2, robot_size[0], robot_size[1]]
    pygame.draw.rect(screen, color, rect, 0)

def draw_state(screen, bars, info_1, info_2):
    draw_init(screen, bars)

    draw_robot(screen, info_1[0][:2], [255,0,0])
    draw_robot(screen, info_1[1][:2], [255,0,0])

    draw_robot(screen, info_2[0][:2], [0,255,0])
    draw_robot(screen, info_2[1][:2], [0,255,0])

    pygame.display.update()