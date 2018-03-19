import numpy as np
import time
import matplotlib.pyplot as plt
import pdb
scale = 0.1
robot_size = [int(500*scale), int(500*scale)]
d_size = [_/2 for _ in robot_size]


class PathPlanning:
    """
    Path Planning Method class
    """
    def __init__(self, cur_pos, dst_pos):
        self.cur_pos = cur_pos
        self.dst_pos = dst_pos

    def update_dst(self, update_pos):
        """
        Update destination position
        :param update_pos: updated destination position
        :return: None
        """
        self.dst_pos = update_pos

    @property
    def get_dst(self):
        """
        Get current destination
        :return: destination coordinate
        """
        return self.dst_pos

    def naive(self, step):
        """
        Naive path planning method. It takes a straight line towards the destination.
        :param step: step length
        :return: a movement in two directions
        """
        return (self.dst_pos - self.cur_pos) / np.linalg.norm(self.dst_pos - self.cur_pos) * step


    def isPos(self, pos, dir_tmp, weight_map):
        """
        judge corners' position whether illegal
        """
        h,w = weight_map.shape
        pos_tmp = (pos[0]+dir_tmp[0], pos[1]+dir_tmp[1])
        corners = [[pos_tmp[0]+d_size[0], pos_tmp[1]+d_size[1]], [pos_tmp[0]+d_size[0], pos_tmp[1]-d_size[1]], [pos_tmp[0]-d_size[0], pos_tmp[1]+d_size[1]], [pos_tmp[0]-d_size[0], pos_tmp[1]-d_size[1]]]
        for corner in corners:
            if corner[0]<0 or corner[0]>=w or corner[1]<0 or corner[1]>=h:
                return False
        if sum(sum(weight_map[pos_tmp[0]-d_size[0]:pos_tmp[0]+d_size[0], pos_tmp[1]-d_size[1]:pos_tmp[1]+d_size[1]])) < 3 * 255 * 4 * d_size[0] * d_size[1]:
           return False
        return True


    def getChildren(self, pos, weight_map, map_father, map_dist):
        """
        get children and update its father and distance
        """
        threshold = 3 * 255
        dir_10 = [(1,0), (-1,0), (0,1), (0,-1)]
        dir_14 = [(1,1), (-1,1), (1,-1), (-1,-1)]

        h,w = weight_map.shape
        pos_children = []
        # up,down,left,right
        for dir_tmp in dir_10:
            if not self.isPos(pos, dir_tmp, weight_map):
                continue
            pos_tmp = (pos[0]+dir_tmp[0], pos[1]+dir_tmp[1])
            if weight_map[pos_tmp]>=threshold and map_dist[pos_tmp]<0:
                map_father[pos_tmp] = pos
                map_dist[pos_tmp] =  map_dist[pos] + 10
                pos_children.append(pos_tmp)

        # left-up,right-down,left-down,right-up
        for dir_tmp in dir_14:
            if not self.isPos(pos, dir_tmp, weight_map):
                continue
            pos_tmp = (pos[0]+dir_tmp[0], pos[1]+dir_tmp[1])
            if weight_map[pos_tmp]>=threshold and map_dist[pos_tmp]<0:
                map_father[pos_tmp] = pos
                map_dist[pos_tmp] =  map_dist[pos] + 14
                pos_children.append(pos_tmp)
        return pos_children


    def astar(self, weight_map):
        """
        simple A-star
        """
        cur_pos = tuple(self.cur_pos)
        dst_pos = tuple(self.dst_pos)
        while len(weight_map.shape)>2:
            weight_map = np.sum(weight_map, -1)
        h,w = weight_map.shape

        # init
        map_father = -1 * np.ones((h,w,2), np.int16)
        map_dist   = -1 * np.ones((h,w), np.int16)
        pos_tbd = [cur_pos]

        # map
        map_father[cur_pos] = cur_pos
        map_dist[cur_pos] = 0
        not_end = True
        while len(pos_tbd)>0 and not_end:
            pos = pos_tbd.pop(0)
            # get child
            pos_children = self.getChildren(pos, weight_map, map_father, map_dist)
            if len(pos_children) > 0:
                pos_tbd += pos_children
                for pos_tmp in pos_children:
                    if pos_tmp[0]==dst_pos[0] and pos_tmp[1]==dst_pos[1]:
                        not_end = False

        # get way
        way = []
        pos_father = dst_pos
        while pos_father[0]!=cur_pos[0] or pos_father[1]!=cur_pos[1]:
            way.append(pos_father)
            # get father
            pos_father = tuple(map_father[pos_father])
        way.append(cur_pos)
        way.reverse()

        '''
        print 'The way from', cur_pos, 'to', dst_pos
        for pos in way:
            print pos
        '''
        #return way, weight_map
        next_pos = cur_pos
        for pos in way:
            if abs(pos[0]-cur_pos[0])>7 or abs(pos[1]-cur_pos[1])>7:
                break
            next_pos = pos
        print 'cur pos', cur_pos, 'dst pos', dst_pos
        print 'next pos', next_pos
        return np.array(next_pos)

    # Algorithm dictionary
    algorithms = {'naive': naive}

    def run(self, cur_pos, step, weight_map, strategy='naive'):
        """
        Run the path planning once
        :param cur_pos: input current position
        :param step: input step length
        :param weight_map: input weight map for path planning
        :param strategy: input destination position
        :return: a movement in two directions
        """
        self.cur_pos = cur_pos
        #act = self.algorithms[strategy](self, step)
        act = self.astar(weight_map)
        #pdb.set_trace()
        act = np.array(np.round(act), dtype=int)
        for i, a in enumerate(act):
            if a < -3:
                act[i] = -3
            elif a > 3:
                act[i] = 3
        ret_act = np.zeros(15, dtype=int)
        ret_act[act[0] + 3] = 1
        ret_act[act[1] + 3 + 7] = 1

        return act#ret_act

    # TODO: Add A-star/Dijkstra


if __name__ == '__main__':
    cur_pos = np.array([30, 30])
    dst_pos = np.array([90, 30])
    pp = PathPlanning(cur_pos, dst_pos)

    weight_map = 255 * np.ones((500,800,3), np.int16)
    for i in range(70):
        weight_map[60,i]=(0,0,0)

    #cur_pos.all()!=dst_pos.all()
    while cur_pos[0]!=dst_pos[0] or cur_pos[1]!=dst_pos[1]:#(cur_pos != dst_pos).all():
        dp = pp.run(cur_pos, 3, weight_map)
        dp = np.array(np.ceil(dp), dtype=int)
        cur_pos += dp
        #pdb.set_trace()
        print cur_pos
