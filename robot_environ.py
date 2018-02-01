import tensorflow as tf
import numpy as np
import random
import pygame
import pdb
scale = 0.1
robot_size = [int(500*scale), int(500*scale)]
act_dict = [-12, -8, -4, 0, 4, 8, 12]
d_size = [_/2 for _ in robot_size]
# Yelly addition:
# bonus-related
RFID_size = [10, 10]
RFID_shift = [0, -10]
RFID_d_size = [_/2 for _ in RFID_size]
bonus_zone_side_len = int(500*scale)
bonus_zone_x_min = int(4000*scale) - bonus_zone_side_len/2 - 1 # notice index begins from 0
bonus_zone_x_max = int(4000*scale) + bonus_zone_side_len/2 - 1 # notice index begins from 0
bonus_zone_y_min = int(2500*scale) - bonus_zone_side_len/2 - 1 # notice index begins from 0
bonus_zone_y_max = int(2500*scale) + bonus_zone_side_len/2 - 1 # notice index begins from 0
time_slice = 0.1 # interval between each decision, in second
bonus_time = 5 # time inside bonus zone to get bonus, in second 
bonus_steps = bonus_time / time_slice # how many continuous steps to take until get bonus

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

# Yelly modification: make these invariables global variables and 
# eliminating all these parameters to functions in this file
raw_map_img, bars = create_raw_map_img()
print 'in robot_environ.py, raw_amp_img initilized'

def draw_pos_tool(pos, color, map_img):
    for i in range(pos[0]-d_size[0], pos[0]+d_size[0]):
        for j in range(pos[1]-d_size[1], pos[1]+d_size[1]):
            map_img[i,j] = color
    return map_img

def draw_pos(pos_1, pos_2):
    map_img = raw_map_img.copy()
    # red
    map_img = draw_pos_tool(pos_1[0][:2], np.array([255, 0, 0],dtype=np.float32), map_img)
    # Yelly comment: why use element 2,3 here rather than 0,1 as above?    
    map_img = draw_pos_tool(pos_1[1][2:], np.array([255, 0, 0],dtype=np.float32), map_img)
    # blue
    map_img = draw_pos_tool(pos_2[0][:2], np.array([0, 255, 0],dtype=np.float32), map_img)
    map_img = draw_pos_tool(pos_2[1][2:], np.array([0, 255, 0],dtype=np.float32), map_img)
    return map_img

# Yelly modification:
# use global variable 'bars' to detect collision with each bar, return dir_new
# notice that a robot could collide into one bar at most,
# so the function could safely return once collision with a bar detected

# I abandoned previous method of bar collision detection because:
# 1. without specific information of bar, caller of this function (i.e. move_detect_wall_collide function) 
#    has to employ while loop to decide dir_new, causing great overhead
# 2. if only use sampled points to detect collision,
#    the quantity of the sampled points should be something large and hard to decide (corners only is definitely not enough)
#    and to cover possible collision during movement, at least three states should be checked(final_pos, move_x_only, move_y_only)
def _move_tool(pos_pre, dir):
    # return dir_new only when collision is confirmed,
    # and that would be inside the loop
    # Yelly declares this variable here just to save per-iteration assignment
    dir_new = dir

    robot_x_min = pos_pre[0] - d_size[0]
    robot_x_max = pos_pre[0] + d_size[0]
    robot_y_min = pos_pre[1] - d_size[1]
    robot_y_max = pos_pre[1] + d_size[1]
    for i in range(len(bars)):
        # per-iteration initialization can be saved if dir_new is properly assigned
        # for real collision        
        #dir_new = dir # re-initialize dir_new at every iteration
        
        # first ensure collision is possible on X axis
        if robot_x_min <= bars[i][0]:
            if robot_x_max > bars[i][0]:
                # collide possible on X axis even without movement
                # if collide, should only adjust movement on Y axis
                dir_new[0] = dir[0]
                pass
            elif robot_x_max + dir[0] > bars[i][0]:
                # collide possible on X axis after movement
                dir_new[0] = bars[i][0] - robot_x_max
            else:
                # collide impossible on X axis
                continue
                
        elif robot_x_min < bars[i][0] + bars[i][2]:
            # collide possible on X axis even without movement
            # if collide, should only adjust movement on Y axis
            dir_new[0] = dir[0]
            pass
        elif robot_x_min + dir[0] < bars[i][0] + bars[i][2]:
            # collide possible on X axis after movement
            dir_new[0] = bars[i][0] + bars[i][2] - robot_x_min
        else:
            # collide impossible on X axis
            continue

        # if collision is possible on X axis, check Y axis
        # if collision is also possible on Y axis, 
        # it's safe to directly return dir_new        
        if robot_y_min <= bars[i][1]:
            if robot_y_max > bars[i][1]:
                # collide possible on Y axis even without movement
                # if collide, should only adjust movement on X axis
                dir_new[1] = dir[1]
                return dir_new
            elif robot_y_max + dir[1] > bars[i][1]:
                # collide possible on Y axis after movement
                dir_new[1] = bars[i][1] - robot_y_max
                return dir_new
            else:
                # collide impossible on Y axis
                continue
                
        elif robot_y_min < bars[i][1] + bars[i][3]:
            # collide possible on Y axis even without movement
            # if collide, should only adjust movement on X axis
            dir_new[1] = dir[1]
            return dir_new
        elif robot_y_min + dir[1] < bars[i][1] + bars[i][3]:
            # collide possible on Y axis after movement
            dir_new[1] = bars[i][1] + bars[i][3] - robot_y_min
            return dir_new
        else:
            # collide impossible on Y axis
            continue

    # no collision detected if function dose not return inside the loop
    return dir


# Yelly addition: this function should be called after move_detect_wall_collide() function,
# that is, (after dir adjustment) movement of robots should not collide into walls.
# move_detect_robot_collide:
# return: collide, adjusted_dir_1, adjusted_dir_2
# if no collision, return previous dirs.
# otherwise return dirs before collision happens
def move_detect_robot_collide(pos_pre_1, pos_pre_2, dir_1, dir_2):
    dir_1_new = dir_1
    dir_2_new = dir_2
    
    safety_dist_x = robot_size[0]
    safety_dist_y = robot_size[1]

    #collide = True

    # possible collide only if the shortest distance (on both X and Y axis) 
    # between points on the two traces is smaller than safety distance
    
    dist_x_pre = abs(pos_pre_2[0] - pos_pre_1[0])
    if (dist_x_pre < safety_dist_x):
        # x distance smaller than safety distance even before movement    
        # no need to change dir[0] here because collision must be caused by Y axis movement
        #collide = True
        pass
    else:
        pos_12_x_sign = _sign(pos_pre_2[0] - pos_pre_1[0])
        dir_1_x_sign = _sign(dir_1[0])
        dir_2_x_sign = _sign(dir_2[0])

        if not dir_1_x_sign == pos_12_x_sign:
            if dir_2_x_sign == pos_12_x_sign:
                # robots move away from each other
                #collide = False
                return False, dir_1, dir_2
            elif abs(dir_2[0]) > dist_x_pre - safety_dist_x:
                # robot_2 moves towards robot_1
                #collide = True
                dir_2_new[0] = _sign(dir_2[0]) * (dist_x_pre - safety_dist_x)
            else:
                # robot_2 moves towards robot_1
                #collide = False
                return False, dir_1, dir_2
        else:
            if dir_2_x_sign == pos_12_x_sign:
                # robot_1 moves towards robot_2, robot_2 move away from robot_1
                if abs(dir_1[0]) > dist_x_pre - safety_dist_x:
                    #collide = True
                    dir_1_new[0] = _sign(dir_1[0]) * (dist_x_pre - safety_dist_x)
                else:
                    #collide = False
                    return False, dir_1, dir_2
            elif abs(dir_1[0] - dir_2[0]) > dist_x_pre - safety_dist_x:
                # robots move towards each other
                #collide = True
                # let the robots come at a equal distance to the middle of there x position
                x_mid = (pos_pre_1[0] + pos_pre_2[0])/2
                dir_1_new[0] = _sign(dir_1[0]) * (abs(x_mid - pos_pre_1[0]) - safety_dist_x/2)
                dir_2_new[0] = _sign(dir_2[0]) * (abs(x_mid - pos_pre_2[0]) - safety_dist_x/2)
            else:
                # robots move towards each other
                #collide = False
                return False, dir_1, dir_2

    # That the function does not return before here, 
    # suggests that collision is possible with regard to X axis
    
    # examine on Y axis below,
    # if collision is also possible with regard to Y axis
    # then collision is reported
    dist_y_pre = abs(pos_pre_2[1] - pos_pre_1[1])
    if (dist_y_pre < safety_dist_y):
        # y distance smaller than safety distance even before movement    
        # no need to change dir[1] here because collision must be caused by X axis movement
        #collide = True
        return True, dir_1_new, dir_2_new
    else:
        pos_12_y_sign = _sign(pos_pre_2[1] - pos_pre_1[1])
        dir_1_y_sign = _sign(dir_1[1])
        dir_2_y_sign = _sign(dir_2[1])

        if not dir_1_y_sign == pos_12_y_sign:
            if dir_2_y_sign == pos_12_y_sign:
                # robots move away from each other
                #collide = False
                return False, dir_1, dir_2
            elif abs(dir_2[1]) > dist_y_pre - safety_dist_y:
                # robot_2 moves towards robot_1
                #collide = True
                dir_2_new[1] = _sign(dir_2[1]) * (dist_y_pre - safety_dist_y)
                return True, dir_1_new, dir_2_new
            else:
                # robot_2 moves towards robot_1
                #collide = False
                return False, dir_1, dir_2
        else:
            if dir_2_y_sign == pos_12_y_sign:
                # robot_1 moves towards robot_2, robot_2 move away from robot_1
                if abs(dir_1[1]) > dist_y_pre - safety_dist_y:
                    #collide = True
                    dir_1_new[1] = _sign(dir_1[1]) * (dist_y_pre - safety_dist_y)
                    return True, dir_1_new, dir_2_new
                else:
                    #collide = False
                    return False, dir_1, dir_2
            elif abs(dir_1[1] - dir_2[1]) > dist_y_pre - safety_dist_y:
                # robots move towards each other
                #collide = True
                # let the robots come at a equal distance to the middle of there y position
                y_mid = (pos_pre_1[1] + pos_pre_2[1])/2
                dir_1_new[1] = _sign(dir_1[1]) * (abs(y_mid - pos_pre_1[1]) - safety_dist_y/2)
                dir_2_new[1] = _sign(dir_2[1]) * (abs(y_mid - pos_pre_2[1]) - safety_dist_y/2)
                return True, dir_1_new, dir_2_new
            else:
                # robots move towards each other
                #collide = False
                return False, dir_1, dir_2

    print '[ERROR]move_detect_robot_collide function has unhandled case!'
    return False, dir_1, dir_2

# Yelly comment: previously dir_new unused. Yelly modified this
def move_detect_wall_collide(pos_pre, dir):
 
    dir_new = dir

    # Yelly modification: change dir to the point before collide (with outbound or bars),
    # rather than use while loop

    # out
    if (pos_pre[0]+_sign(dir[0])*(abs(dir[0])+d_size[0])) < 0:
        dir_new[0] = d_size[0] - pos_pre[0] # negative
    elif (pos_pre[0]+_sign(dir[0])*(abs(dir[0])+d_size[0]))>=int(scale*8000):
        # notice: biggest index is N-1
        dir_new[0] = (int(scale*8000)-1) - d_size[0] - pos_pre[0] # positive
    
    if (pos_pre[1]+_sign(dir[1])*(abs(dir[1])+d_size[1])) < 0:
        dir_new[1] = d_size[1] - pos_pre[1] # negative
    elif (pos_pre[1]+_sign(dir[1])*(abs(dir[1])+d_size[1]))>=int(scale*5000):
        # notice: biggest index is N-1
        dir_new[1] = (int(scale*5000)-1) - d_size[1] - pos_pre[1] # positive

    # bar
    print 'move to:', [pos_pre[0]+_sign(dir_new[0])*(abs(dir_new[0])+d_size[0]), pos_pre[1]+_sign(dir_new[1])*(abs(dir_new[1])+d_size[1])],

    #print sum(raw_map_img[pos_pre[0]+_sign(dir_new[0])*(abs(dir_new[0])+d_size[0]), pos_pre[1]]), sum(raw_map_img[pos_pre[0], pos_pre[1]+_sign(dir_new[1])*(abs(dir_new[1])+d_size[1])])
    print 'dir_inside_bound',dir_new,

    # Yelly modification:
    # _move_tools() function returns dir_new
    dir_new = _move_tool(pos_pre, dir_new)
    print 'dir_new',dir_new
    return dir_new

# Yelly comment: [TODO]need much modifications on this function!
def shoot(pos, target, state_1, state_2, map_img):
    flag = False
    dx   = int(robot_size[0]/10)
    dy   = int(robot_size[0]/10) # Yelly comment: should be robot_size[1]/10 ?
    targets = [[target[0]+dx, target[1]+dy], [target[0]+dx, target[1]-dy], [target[0]-dx, target[1]+dy], [target[0]-dx, target[1]-dy]]
    for tar in targets:
        for tmp_x in range(pos[0], tar[0]):
            tmp_y = pos[1] + int((tar[1]-pos[1])/(tar[0]-pos[0])) * (tmp_x-pos[0])
            if sum(map_img[tmp_x, tmp_y]) < 2 * 255:
                return False, state_1, state_2
    state_1['hit'] = True
    state_2['shooted'] = True
    return True, state_1, state_2

# Yelly comment: [TODO] action selection (according to policy) should be put into agent logic
def get_action(act, hp, policy='MAX'):
    # prevent dead robots from action
    if hp <= 0:
        return [0, 0, 0]

    if policy=='MAX':
        return [act_dict[act[0]], act_dict[act[1]], act[-1]]
    #return [act_dict[random.randint(0,6)], act_dict[random.randint(0,6)], act_dict[random.randint(0,1)]]
    return [act_dict[0], act_dict[0], act_dict[random.randint(0,1)]]


def get_state(info_1, info_2, act_1, act_2, policy='MAX'):
    # get state[x,y,dx,dy,shoot,hit,shooted,WallCollide,TeamCollide,AICollide]
    state_1 = []
    state_2 = []
    # Yelly modification: pass hp info to get_action function to prevent dead robots from action
    act_1 = [get_action(act_1[0], info_1[0][2], policy=policy), get_action(act_1[1], info_1[1][2], policy=policy)]
    act_2 = [get_action(act_2[0], info_2[0][2], policy=policy), get_action(act_2[1], info_2[1][2], policy=policy)]


    # move:[dx,dy]
    # Yelly comment: divide collision cases into three clases:
    # 1. [WallCollide] team robot collide into bar/fence
    # 2. [TeamCollide] team robot collide into each other
    # 3. [AICollide] team robot collide into AI robot
    # For all these collisions, the actions take the robots to the point before collision
    # if dir_tmp different from act, suggesting collision detected
    
    # WallCollide
    # prevent dead robots from moving itself
    if info_1[0][2] > 0:
        dir_tmp = move_detect_wall_collide(info_1[0][:2], act_1[0][:2])
        state_1.append({'x':info_1[0][0], 'y':info_1[0][1], 'dx':dir_tmp[0], 'dy':dir_tmp[1]})
        if ( not dir_tmp[0] == act_1[0][0] or not dir_tmp[1] == act_1[0][1]):
            print 'robot 0 of team 1 collide detected',
            print 'dir_tmp[0]='+str(dir_tmp[0])+',act_1[0][0]='+str(act_1[0][0])+',dir_tmp[1]='+str(dir_tmp[1])+',act_1[0][1]='+str(act_1[0][1])
            state_1[0]['WallCollide'] = True
        else:
            state_1[0]['WallCollide'] = False
    if info_1[1][2] > 0:
        dir_tmp = move_detect_wall_collide(info_1[1][:2], act_1[1][:2])
        state_1.append({'x':info_1[1][0], 'y':info_1[1][1], 'dx':dir_tmp[0], 'dy':dir_tmp[1]})
        if ( not dir_tmp[0] == act_1[1][0] or not dir_tmp[1] == act_1[1][1]):
            print 'robot 1 of team 1 collide detected',
            print 'dir_tmp[0]='+str(dir_tmp[0])+',act_1[1][0]='+str(act_1[1][0])+',dir_tmp[1]='+str(dir_tmp[1])+',act_1[1][1]='+str(act_1[1][1])
            state_1[1]['WallCollide'] = True
        else:
            state_1[1]['WallCollide'] = False

    if info_2[0][2] > 0:
        dir_tmp = move_detect_wall_collide(info_2[0][:2], act_2[0][:2])
        state_2.append({'x':info_2[0][0], 'y':info_2[0][1], 'dx':dir_tmp[0], 'dy':dir_tmp[1]})
        if ( not dir_tmp[0] == act_2[0][0] or not dir_tmp[1] == act_2[0][1]):
            print 'robot 0 of team 2 collide detected',
            print 'dir_tmp[0]='+str(dir_tmp[0])+',act_2[0][0]='+str(act_2[0][0])+',dir_tmp[1]='+str(dir_tmp[1])+',act_2[0][1]='+str(act_2[0][1])
            state_2[0]['WallCollide'] = True
        else:
            state_2[0]['WallCollide'] = False
    if info_2[0][2] > 0:
        dir_tmp = move_detect_wall_collide(info_2[1][:2], act_2[1][:2])
        state_2.append({'x':info_2[1][0], 'y':info_2[1][1], 'dx':dir_tmp[0], 'dy':dir_tmp[1]})
        if ( not dir_tmp[0] == act_2[1][0] or not dir_tmp[1] == act_2[1][1]):
            print 'robot 1 of team 2 collide detected',
            print 'dir_tmp[0]='+str(dir_tmp[0])+',act_2[1][0]='+str(act_2[1][0])+',dir_tmp[1]='+str(dir_tmp[1])+',act_2[1][1]='+str(act_2[1][1])
            state_2[1]['WallCollide'] = True
        else:
            state_2[1]['WallCollide'] = False

    # [TODO] cases where one of two robots is dead
    # TeamCollide
    collide, dir_tmp_1, dir_tmp_2 = move_detect_robot_collide(info_1[0][:2], info_1[1][:2], [state_1[0]['dx'], state_1[0]['dy']], [state_1[1]['dx'], state_1[1]['dy']])
    if (collide):
        print 'team 1 robots (prev pos = ', [info_1[0][0], info_1[0][1]], [info_1[1][0], info_1[1][1]], ') collide with each other, change dir from ',
        print [state_1[0]['dx'], state_1[0]['dy']], [state_1[1]['dx'], state_1[1]['dy']], ' to ', dir_tmp_1, dir_tmp_2
        state_1[0]['TeamCollide'] = True
        state_1[1]['TeamCollide'] = True
        state_1[0]['dx'] = dir_tmp_1[0]
        state_1[0]['dy'] = dir_tmp_1[1]
        state_1[1]['dx'] = dir_tmp_2[0]
        state_1[1]['dy'] = dir_tmp_2[1]
    else:
        state_1[0]['TeamCollide'] = False
        state_1[1]['TeamCollide'] = False

    collide, dir_tmp_1, dir_tmp_2 = move_detect_robot_collide(info_2[0][:2], info_2[1][:2], [state_2[0]['dx'], state_2[0]['dy']], [state_2[1]['dx'], state_2[1]['dy']])
    if (collide):
        print 'team 2 robots (prev pos = ', [info_2[0][0], info_2[0][1]], [info_2[1][0], info_2[1][1]], ') collide with each other, change dir from ',
        print [state_2[0]['dx'], state_2[0]['dy']], [[state_2[1]['dx']], state_2[1]['dy']], ' to ', dir_tmp_1, dir_tmp_2
        state_2[0]['TeamCollide'] = True
        state_2[1]['TeamCollide'] = True
        state_2[0]['dx'] = dir_tmp_1[0]
        state_2[0]['dy'] = dir_tmp_1[1]
        state_2[1]['dx'] = dir_tmp_2[0]
        state_2[1]['dy'] = dir_tmp_2[1]
    else:
        state_2[0]['TeamCollide'] = False
        state_2[1]['TeamCollide'] = False

    # AICollide
    collide, dir_tmp_1, dir_tmp_2 = move_detect_robot_collide(info_1[0][:2], info_2[0][:2], [state_1[0]['dx'], state_1[0]['dy']], [state_2[0]['dx'], state_2[0]['dy']])
    if (collide):
        print 'team 1 robot 0 collide with team 2 robot 0, change dir from ',
        print [state_1[0]['dx'], state_1[0]['dy']], [state_2[0]['dx'], state_2[0]['dy']], ' to ', dir_tmp_1, dir_tmp_2
        state_1[0]['AICollide'] = True
        state_2[0]['AICollide'] = True
        state_1[0]['dx'] = dir_tmp_1[0]
        state_1[0]['dy'] = dir_tmp_1[1]
        state_2[0]['dx'] = dir_tmp_2[0]
        state_2[0]['dy'] = dir_tmp_2[1]
    else:
        state_1[0]['AICollide'] = False
        state_2[0]['AICollide'] = False

    collide, dir_tmp_1, dir_tmp_2 = move_detect_robot_collide(info_1[0][:2], info_2[1][:2], [state_1[0]['dx'], state_1[0]['dy']], [state_2[1]['dx'], state_2[1]['dy']])
    if (collide):
        print 'team 1 robot 0 collide with team 2 robot 1, change dir from ',
        print [state_1[0]['dx'], state_1[0]['dy']], [state_2[1]['dx'], state_2[1]['dy']], ' to ', dir_tmp_1, dir_tmp_2
        state_1[0]['AICollide'] = True
        state_2[1]['AICollide'] = True
        state_1[0]['dx'] = dir_tmp_1[0]
        state_1[0]['dy'] = dir_tmp_1[1]
        state_2[1]['dx'] = dir_tmp_2[0]
        state_2[1]['dy'] = dir_tmp_2[1]
    else:
        state_1[0]['AICollide'] = False
        state_2[1]['AICollide'] = False

    collide, dir_tmp_1, dir_tmp_2 = move_detect_robot_collide(info_1[1][:2], info_2[0][:2], [state_1[1]['dx'], state_1[1]['dy']], [state_2[0]['dx'], state_2[0]['dy']])
    if (collide):
        print 'team 1 robot 1 collide with team 2 robot 0, change dir from ',
        print [state_1[1]['dx'], state_1[1]['dy']], [state_2[0]['dx'], state_2[0]['dy']], ' to ', dir_tmp_1, dir_tmp_2
        state_1[1]['AICollide'] = True
        state_2[0]['AICollide'] = True
        state_1[1]['dx'] = dir_tmp_1[0]
        state_1[1]['dy'] = dir_tmp_1[1]
        state_2[0]['dx'] = dir_tmp_2[0]
        state_2[0]['dy'] = dir_tmp_2[1]
    else:
        state_1[1]['AICollide'] = False
        state_2[0]['AICollide'] = False

    collide, dir_tmp_1, dir_tmp_2 = move_detect_robot_collide(info_1[1][:2], info_2[1][:2], [state_1[1]['dx'], state_1[1]['dy']], [state_2[1]['dx'], state_2[1]['dy']])
    if (collide):
        print 'team 1 robot 1 collide with team 2 robot 1, change dir from ',
        print [state_1[1]['dx'], state_1[1]['dy']], [state_2[1]['dx'], state_2[1]['dy']], ' to ', dir_tmp_1, dir_tmp_2
        state_1[1]['AICollide'] = True
        state_2[1]['AICollide'] = True
        state_1[1]['dx'] = dir_tmp_1[0]
        state_1[1]['dy'] = dir_tmp_1[1]
        state_2[1]['dx'] = dir_tmp_2[0]
        state_2[1]['dy'] = dir_tmp_2[1]
    else:
        state_1[1]['AICollide'] = False
        state_2[1]['AICollide'] = False

    # shoot init
    if act_1[0][-1]:
        state_1[0]['shoot']= True
    else:
        state_1[0]['shoot']= False
    state_1[0]['hit']      = False
    state_1[0]['shooted']  = False
    if act_1[1][-1]:
        state_1[1]['shoot']= True
    else:
        state_1[1]['shoot']= False
    state_1[1]['hit']      = False
    state_1[1]['shooted']  = False

    if act_2[0][-1]:
        state_2[0]['shoot']= True
    else:
        state_2[0]['shoot']= False
    state_2[0]['hit']      = False
    state_2[0]['shooted']  = False
    if act_2[1][-1]:
        state_2[1]['shoot']= True
    else:
        state_2[1]['shoot']= False
    state_2[1]['hit']      = False
    state_2[1]['shooted']  = False
    
    # Yelly comment: 
    # [TODO] parameter 'map_img' passed to shoot() function should not be raw_map_img
    # but the map_img reflecting robots,
    # However as shoot() function will be modified further,
    # now I just pass raw_map_img to shoot() function (as is did by Yongzhuang)
    # hit or shooted
    if state_1[0]['shoot']:
        shoot(info_1[0][:2], info_2[0][:2], state_1[0], state_2[0], raw_map_img)
        shoot(info_1[0][:2], info_2[1][:2], state_1[0], state_2[1], raw_map_img)
    if state_1[1]['shoot']:
        shoot(info_1[1][:2], info_2[0][:2], state_1[1], state_2[0], raw_map_img)
        shoot(info_1[1][:2], info_2[1][:2], state_1[1], state_2[1], raw_map_img)

    if state_2[0]['shoot']:
        shoot(info_2[0][:2], info_1[0][:2], state_2[0], state_1[0], raw_map_img)
        shoot(info_2[0][:2], info_1[1][:2], state_2[0], state_1[1], raw_map_img)
    if state_2[1]['shoot']:
        shoot(info_2[1][:2], info_1[0][:2], state_2[1], state_1[0], raw_map_img)
        shoot(info_2[1][:2], info_1[1][:2], state_2[1], state_1[1], raw_map_img)
   

    return state_1, state_2

# Yelly modification: prevent dead robots from getting reward
def _get_reward(state, my_bonus_cnt, oppo_bonus_cnt, hp):
    if hp == 0:
        state['reward'] = 0
        return state

    # move
    state['reward'] = 3 + np.sqrt((state['x']+state['dx']-4000*scale)**2 + (state['y']+state['dy']-2500*scale)**2) - np.sqrt((state['x']-4000*scale)**2 + (state['y']-2500*scale)**2)

    # shoot, hit or shooted
    if state['hit']:
        if my_bonus_cnt == bonus_steps:
            state['reward'] += 15. 
        else:
            state['reward'] += 10.
    elif state['shoot'] and not state['hit']:
        state['reward'] += -1.

    if state['shooted']:
        if oppo_bonus_cnt == bonus_steps:
            state['reward'] += -15. 
        else:
            state['reward'] += -10.

    # Yelly addition:
    # collision
    #wall_collide_reward = -15 # configurable, Yelly make it no less than the max move reward I may get
    #team_collide_reward = -10 # configurable, Yelly make it a little less than wall_collide_reward
    #ai_collide_reward = 0 # configurable, Yelly set it 0 here, because there's no obvious damage caused by run into opponent (I heard that our robots will be even stronger than DJI robots). However, if the distance is too small hit rate might decrease.
    if state['WallCollide']:
        state['reward'] += -15.
    if state['TeamCollide']:
        state['reward'] += -10.
    #if state['AICollide']:
    #    state['reward'] += 0.

    return state

def get_reward(info_1, info_2, act_1, act_2, policy='MAX'):
    # init
    state_1, state_2 = get_state(info_1, info_2, act_1, act_2, policy=policy)
    reward_1 = np.zeros([2,15])
    reward_2 = np.zeros([2,15])

    # reward
    state_1 = [_get_reward(state_1[0], info_1[0][3], info_2[0][3], info_1[0][2]), _get_reward(state_1[1], info_1[0][3], info_2[0][3], info_1[1][2])]
    state_2 = [_get_reward(state_2[0], info_2[0][3], info_1[0][3], info_2[0][2]), _get_reward(state_2[1], info_2[0][3], info_1[0][3], info_2[1][2])]

    reward_1[0,act_1[0][0]] = state_1[0]['reward']
    reward_1[0,act_1[0][1]+7] = state_1[0]['reward']
    if state_1[0]['hit']:
        reward_1[0,-1] = state_1[0]['reward']
    reward_1[1,act_1[0][0]] = state_1[1]['reward']
    reward_1[1,act_1[0][1]+7] = state_1[1]['reward']
    if state_1[1]['hit']:
        reward_1[1,-1] = state_1[1]['reward']

    reward_2[0,act_2[0][0]] = state_2[0]['reward']
    reward_2[0,act_2[0][1]+7] = state_2[0]['reward']
    if state_2[0]['hit']:
        reward_2[0,-1] = state_2[0]['reward']
    reward_2[1,act_2[0][0]] = state_2[1]['reward']
    reward_2[1,act_2[0][1]+7] = state_2[1]['reward']
    if state_2[1]['hit']:
        reward_2[1,-1] = state_2[1]['reward']

    return state_1, state_2, reward_1, reward_2

# Yelly addition:
# if after the action taken, the RFID module is completely inside bonus zone, return True
# otherwise, return False
#
# [TODO]better identify bonus zone by different color on raw map
# because it is safe to judge whether four corners of RFID module is inside bonus zone
# and it's easier adapt this method to the cases where rotation is considered
def inside_bonus_zone(pos):
    module_x_min = pos[0] + RFID_shift[0] - RFID_d_size[0]
    if module_x_min < bonus_zone_x_min:
        return False

    module_x_max = pos[0] + RFID_shift[0] + RFID_d_size[0]
    if module_x_max > bonus_zone_x_max:
        return False

    module_y_min = pos[1] + RFID_shift[1] - RFID_d_size[1]
    if module_y_min < bonus_zone_y_min:
        return False

    module_y_max = pos[1] + RFID_shift[1] + RFID_d_size[1]
    if module_y_max > bonus_zone_y_max:
        return False

    return True

# Yelly modification:
# let this function compute new infos for two robots of one side
# because info[3] (buff) values of the two robots correlate
def get_new_info(info_pre, state):
    # robot 0 info[:3]
    info_pre[0][0] += state[0]['dx']
    info_pre[0][1] += state[0]['dy']
    # Yelly modification:
    # 1. added effect from bonus
    # 2. each projectile hit reduces 50 HP, max projectile frequency is 10 Hz, time slice is 0.1s.
    #	Thus, when shooted, on average robot is hit by one projectile in one time slice.
    #	[TODO] the time interval between shoot action from the opponent taken and projectile hit me
    if state[0]['shooted']:
        if info_pre[0][3] == bonus_steps:
            info_pre[0][2] -= 75
        else:
            info_pre[0][2] -= 50

        if info_pre[0][2] < 0:
            info_pre[0][2] = 0      

    # robot 1 info[:3]
    info_pre[1][0] += state[1]['dx']
    info_pre[1][1] += state[1]['dy']
    if state[1]['shooted']:
        if info_pre[1][3] == bonus_steps:
            info_pre[1][2] -= 75
        else:
            info_pre[1][2] -= 50

        if info_pre[0][2] < 0:
            info_pre[0][2] = 0      
   
    # Yelly addition:
    # bonus 
    # notice that info_pre[:3] has been adapted to the new values above   
    # once one robot get bonus, both robot buff value should equal to bonus_steps
    if info_pre[0][3] < bonus_steps: 
        # robot 0
        if not inside_bonus_zone(info_pre[0][:2]):
            info_pre[0][3] = -1
        else:
            info_pre[0][3] += 1
            if info_pre[0][3] == bonus_steps:
                info_pre[1][3] = bonus_steps
                return info_pre
        # robot 1
        if not inside_bonus_zone(info_pre[1][:2]):
            info_pre[1][3] = -1
        else:
            info_pre[1][3] += 1
            if info_pre[1][3] == bonus_steps:
                info_pre[0][3] = bonus_steps
                return info_pre
    
    return info_pre

def get_init():
    # Yelly modification: 
    # 1. change initial buff value to -1
    # 	-1 means no accumulating buff residence before
    # 	wlhen robot reside in bonus zone for a period of time,
    # 	add buff value one by one until it achieves bonus_step (5/0.1),
    # 	which indicate that bonus got
    # 2. change initial HP value to 2000
    #info_1 = [[30, 420, 2000, -1],  [30, 470, 2000, -1]]
    #info_2 = [[770, 420, 2000, -1], [770, 470, 2000, -1]]

    # Yelly test
    info_1 = [[375, 225, 2000, -1],  [375, 280, 2000, -1]]
    info_2 = [[425, 225, 2000, -1], [425, 280, 2000, -1]]

    map_img_new = draw_pos(info_1, info_2)
    return info_1, info_2, map_img_new

def environ(flag, info_1, info_2, act_1, act_2, policy='MAX'):
    # info[2,4]: [x,y,blood,buff]
    # act[2,15]:  [3,2,1,0,-1,-2,-3,3,2,1,0,-1,-2,-3,shoot]

    # state and reward
    state_1, state_2, reward_1, reward_2 = get_reward(info_1, info_2, act_1, act_2, policy=policy)
    #pdb.set_trace()

    print '\nstate:'
    print state_1
    print state_2

    # new info and map
# Yelly modification:
# let get_new_info() function compute for two robots at a time
# because buff value is correlated for two team robots
# i.e. if any one of the team robots gets bonus, the two robots of the team should all get bonus
# and info[3] value of the two robots should all be 5/0.1
#
# [TODO] Yelly raises a quesion here:
# If AI robots cannot get bonus at all,
# then the HP drop for one hit on team robot and AI robot could be different,
# so for AI robot it is meaningless to reside in bonus zone for 5s.
# This makes the two neural networks asymmetric.
# Possible solutions include:
# 1. do some tricks in reward computing of each side
# 2. Allow asymmetric, i.e. train only one nn as team robot while the other as AI robot
    info_1_new = get_new_info(info_1, state_1)
    info_2_new = get_new_info(info_2, state_2)

    map_img_new = draw_pos(info_1_new, info_2_new)


    return info_1_new, info_2_new, reward_1, reward_2, map_img_new
