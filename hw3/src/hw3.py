import numpy as np
from PIL import Image
import habitat_sim
from habitat_sim.utils.common import d3_40_colors_rgb
import cv2
import math
import mpmath
import open3d as o3d
import matplotlib.pyplot as plt
import copy
import csv
import scipy.linalg
import json
from load import transform_rgb_bgr, transform_depth, transform_semantic, make_simple_cfg, navigateAndSee
import glob



def load_point_cloud(points_path, colors_path):

    pcd_points = np.load(points_path)   # shape = (n, 3) 
    pcd_colors = np.load(colors_path)   # shape = (n, 3) 


    # rescale
    pcd_points = pcd_points * 10000. / 255.
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_points)
    pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
    # o3d.visualization.draw_geometries([pcd])

    
    return pcd

def remove_ceiling_and_floor(pcd):

    global x_max, z_max, x_min, z_min
    pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:, 1] < 0.13)[0])  # remove ceiling 
    pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:, 1] > -1.15)[0])   # remove floor

    x_max = np.max(np.asarray(pcd.points)[:,0])
    z_max = np.max(np.asarray(pcd.points)[:,2])
    x_min = np.min(np.asarray(pcd.points)[:,0])
    z_min = np.min(np.asarray(pcd.points)[:,2])
    

    # o3d.visualization.draw_geometries([pcd, min_max_point_set, x_axis_set])

    return pcd
def plot_scatter_graph(pcd):
    pcd_points = np.asarray(pcd.points)
    pcd_colors = np.asarray(pcd.colors)

    plt.scatter(pcd_points[:,2], pcd_points[:,0], s=1, c=pcd_colors)
    plt.axis('equal')
    plt.show()
    # plt.savefig('map.png')

def find_pixel_voxel_scale(map, color_set):
    global x_max_map, z_max_map, x_min_map, z_min_map, x_scale, z_scale

    x_max_map = -np.inf
    z_max_map = -np.inf
    x_min_map = np.inf
    z_min_map = np.inf

    for i in range(map.shape[0]):

        if(i<80):
            continue
        if(i>410):
            continue
        for j in range(map.shape[1]):
            if(j<90):
                continue
            if(j>5600):
                continue

            if(tuple(map[i, j]) in color_set):
                if(i > x_max_map):
                    x_max_map = i
                if(j > z_max_map):
                    z_max_map = j
                if(i < x_min_map):
                    x_min_map = i
                if(j < z_min_map):
                    z_min_map = j


    # print(z_max_map, x_max_map, z_min_map, x_min_map)
    # print(z_max, x_min, z_min, x_max)

    # pixel to voxel
    z_scale = (z_max-z_min)/(z_max_map-z_min_map)
    x_scale = (x_max-x_min)/(x_max_map-x_min_map)

    z_max_temp = (z_max_map-z_min_map)*z_scale + z_min
    z_min_temp = (z_min_map-z_min_map)*z_scale + z_min
    x_max_temp = (x_max_map-x_min_map)*(-x_scale) + x_max
    x_min_temp = (x_min_map-x_min_map)*(-x_scale) + x_max
    # print(z_max_temp, x_max_temp, z_min_temp, x_min_temp)

    # print(pixel_to_voxel(z_max_point))
    # print(pixel_to_voxel(x_max_point))
    # print(pixel_to_voxel(z_min_point))
    # print(pixel_to_voxel(x_min_point))
    
    # return z_min_map, z_scale, z_min, x_min_map, x_scale, x_max

def pixel_to_voxel(pixel):
    voxel = np.array([0, 1.5, 0])
    voxel_x = (pixel[0]-x_min_map)*(-x_scale) + x_max
    voxel_z = (pixel[1]-z_min_map)*z_scale + z_min

    return voxel_x, voxel_z

    
def read_color_code():
    path = './color_coding_semantic_segmentation_classes.csv'
    color_code = {}
    color_label = {}
    color_set = set()

    f = open(path, 'r')
    rows = csv.reader(f, delimiter=',')
    for i, row in enumerate(rows):
        # print(row[0])
        if i==0:
            continue
        label = int(row[0])
        r, g, b = row[1].split(',')
        r = r.replace('(', '')
        g = g.replace(' ', '')
        b = b.replace(' ', '')
        b = b.replace(')', '')
        color_label[row[4]] = label
        color_code[row[4]] = tuple([float(b), float(g), float(r)])
        color_set.add(tuple([float(b), float(g), float(r)]))

    
    return color_code, color_set, color_label

def find_target(color_code, map):
    # items = ['cooktop']
    items = ['refrigerator', 'rack', 'cushion', 'lamp', 'cooktop']
    # items = ['refrigerator', 'rack', 'cushion', 'cooktop']
    mass_centers = {'refrigerator': np.array([0, 0]), 
                    'rack': np.array([0, 0]), 
                    'cushion': np.array([0, 0]), 
                    'lamp': np.array([0, 0]), 
                    'cooktop': np.array([0, 0])}
    directions = {'refrigerator': np.array([1, 0]), 
                    'rack': np.array([0, -1]), 
                    'cushion': np.array([-1, -1]), 
                    'lamp': np.array([0, -1]), 
                    'cooktop': np.array([-1, 0])}   
     
    stepsizes = {'refrigerator': 15, 
                    'rack': 15, 
                    'cushion': 25, 
                    'lamp': 15, 
                    'cooktop': 25} 

    targets = {'refrigerator': np.array([0, 0]), 
                    'rack': np.array([0, 0]), 
                    'cushion': np.array([0, 0]), 
                    'lamp': np.array([0, 0]), 
                    'cooktop': np.array([0, 0])}
  
    
    for item in items:

        code = color_code[item]
        code = np.asarray(code)
        code = code.astype(int)
        points_num = 0

        mass_center = np.array([0, 0])
        # map[i, j] -> [b, g, r]
        # code -> [b, g, r]
        for i in range(map.shape[0]):
            for j in range(map.shape[1]):
                if((map[i][j] == code).all()):
                    mass_center = mass_center + np.array([i, j])
                    points_num = points_num + 1

        mass_center = mass_center / points_num
        mass_center = mass_center.astype(int)
        mass_centers[item] = mass_center
        targets[item] = mass_center + directions[item] * stepsizes[item]

        # cv2.circle(map_show, (mass_center[1], mass_center[0]), 2, (0,0,0), 2)
        # cv2.circle(map_show, (targets[item][1], targets[item][0]), 2, (0,0,0), 2)
        # cv2.imshow('map', map_show)

    return targets


# function to display the coordinates of
# of the points clicked on the image
def click_event(event, x, y, flags, params):

    global start_x, start_y
 
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
 
        start_x = x
        start_y = y

        print(map[y, x])

        # displaying the point on the image
        # cv2.circle(影像, 圓心座標, 半徑, 顏色, 線條寬度)
        cv2.circle(map_show, (x, y), 3, (0,255,0), 3)
        

        cv2.imshow('map', map_show)

    # checking for left mouse clicks
    if event == cv2.EVENT_MBUTTONDOWN:
        
        target_x = x
        target_y = y
        print(map[y, x])
        # displaying the point on the image
        cv2.circle(map_show, (x, y), 3, (255,0,0), 3)
        
        cv2.imshow('map', map_show)

def find_nearest_node(node):

    global RRTtree
    
    min_dis = np.inf
    for i in range(RRTtree.shape[0]):
        dis = np.linalg.norm(RRTtree[i, 0:2]-node)
        if dis < min_dis:
            min_dis = dis
            nearest_id = i

    nearest_node = RRTtree[nearest_id, 0:2]
    return nearest_node, nearest_id, min_dis

def obstacle_free(i, j):
    if map[i, j] != np.array([255, 255, 255]):
        return True
    else:
        return False


def check_obstacle_free(n0, n1):
    # n0 -> nearest_node
    # n1 -> new_node
    dir = math.atan2(n1[0]-n0[0], n1[1]-n0[1])

    # print(np.arange(0, np.linalg.norm(n1-n0), 0.5))

    for r in np.arange(0, np.linalg.norm(n1-n0), 0.5):
        
        check_node = n0 + r * np.array([math.sin(dir), math.cos(dir)])

        free1 = (map[int(np.ceil(check_node[0])), int(np.ceil(check_node[1]))]==np.array([255, 255, 255])).all()
        free2 = (map[int(np.ceil(check_node[0])), int(np.floor(check_node[1]))]==np.array([255, 255, 255])).all()
        free3 = (map[int(np.floor(check_node[0])), int(np.ceil(check_node[1]))]==np.array([255, 255, 255])).all()
        free4 = (map[int(np.floor(check_node[0])), int(np.floor(check_node[1]))]==np.array([255, 255, 255])).all()

        if((not free1) or (not free2) or (not free3) or (not free4)):
            return False
        
    free1 = (map[int(np.ceil(n1[0])), int(np.ceil(n1[1]))]==np.array([255, 255, 255])).all()
    free2 = (map[int(np.ceil(n1[0])), int(np.floor(n1[1]))]==np.array([255, 255, 255])).all()
    free3 = (map[int(np.floor(n1[0])), int(np.ceil(n1[1]))]==np.array([255, 255, 255])).all()
    free4 = (map[int(np.floor(n1[0])), int(np.floor(n1[1]))]==np.array([255, 255, 255])).all()

    if((not free1) or (not free2) or (not free3) or (not free4)):
        return False

    return True


def RRT():

    global RRTtree, path

    RRTtree = np.array([[start_y, start_x, -1]])    # y -> row, x -> column, index of parent (start point is the root, no parent, parent id = -1) 
    RRTtree = RRTtree.astype(int)  
    threshold = 20                                  # nodes closer than this threshold are taken as almost the same
    fail_attemp = 0
    max_fail_attemp = 1500
    path_found = False
    path = []

    cv2.circle(map_show, (start_x, start_y), 3, (0,255,0), 3)
    cv2.circle(map_show, (target_x, target_y), 3, (255,0,0), 3)

    # while fail_attemp <= max_fail_attemp:
    for i in range(max_fail_attemp):

        print(i)
        # with probability = 0.3 to pick the target
        if np.random.rand() < 0.7:
            rand_node =  np.multiply(np.random.rand(1, 2)[0], np.array([map.shape[0], map.shape[1]]))      # map.shape = (480, 640, 3)
            rand_node = rand_node.astype(int)
        else:
            rand_node = np.array([target_y, target_x])
            rand_node = rand_node.astype(int)

        # select the node in the RRT tree that is closest to the random node
        nearest_node, nearest_id, nearest_dis = find_nearest_node(rand_node)

        # print(rand_node, nearest_node)


        # # move from the nearest node an incremental distance toward the direction of the random node
        # if(np.linalg.norm(nearest_node-rand_node) <= 20):
        #     new_node = rand_node
        # else: 
        #     step_size = 20
        #     theta = math.atan2(rand_node[0]-nearest_node[0], rand_node[1]-nearest_node[1])
        #     new_node = nearest_node + step_size * np.array([math.sin(theta), math.cos(theta)])
        #     new_node = new_node.astype(int)

        step_size = 20
        theta = math.atan2(rand_node[0]-nearest_node[0], rand_node[1]-nearest_node[1])
        new_node = nearest_node + step_size * np.array([math.sin(theta), math.cos(theta)])
        new_node = new_node.astype(int)

        # check obstacle free
        if(not check_obstacle_free(nearest_node, new_node)):
            # print('fail')
            fail_attemp = fail_attemp + 1
            continue
        

        # # check if new node is already existing in the tree
        # node, id, min_dis = find_nearest_node(new_node)
        # if(min_dis>20):
        #     print('*')
        #     RRTtree = np.append(RRTtree, np.array([[new_node[0], new_node[1], nearest_id]]), axis=0)

        #     # cv2.line(影像, 開始座標, 結束座標, 顏色, 線條寬度)
        #     cv2.line(map_show, (new_node[1], new_node[0]), (nearest_node[1], nearest_node[0]), (0, 0, 0), 1)
        #     # cv2.circle(影像, 圓心座標, 半徑, 顏色, 線條寬度)
        #     cv2.circle(map_show, (new_node[1], new_node[0]), 2, (150,0,255), 1)
        #     cv2.imshow('map', map_show)
        # else: 
        #     # new node is already existing in the tree
        #     fail_attemp = fail_attemp + 1
            
    

        RRTtree = np.append(RRTtree, np.array([[new_node[0], new_node[1], nearest_id]]), axis=0)

        # cv2.line(影像, 開始座標, 結束座標, 顏色, 線條寬度)
        cv2.line(map_show, (new_node[1], new_node[0]), (nearest_node[1], nearest_node[0]), (0, 0, 0), 1)
        # cv2.circle(影像, 圓心座標, 半徑, 顏色, 線條寬度)
        cv2.circle(map_show, (new_node[1], new_node[0]), 2, (255,200,0), 1)
        cv2.imshow('map', map_show)

        # check whether a path to goal is found
        # select the node in the RRT tree that is closest to the random node
        target_node = np.array([target_y, target_x])
        nearest_node, nearest_id, nearest_dis = find_nearest_node(target_node)
        if(nearest_dis < 20):

            if(check_obstacle_free(nearest_node, target_node)):
                print('find path!')
                path_found = True
                RRTtree = np.append(RRTtree, np.array([[target_node[0], target_node[1], nearest_id]]), axis=0)
                cv2.line(map_show, (target_node[1], target_node[0]), (nearest_node[1], nearest_node[0]), (0, 0, 0), 1)
                # cv2.circle(map_show, (target_node[1], target_node[0]), 2, (150,0,255), 1)
                cv2.imshow('map', map_show)
                
                break

    if(path_found):
        node_id = RRTtree.shape[0] - 1      # target id
        cv2.circle(map_show, (target_node[1], target_node[0]), 3, (0,0,255), -1)
        path.append([target_node[0], target_node[1]])

        while True:
            parent_id = RRTtree[node_id][2]

            if(parent_id == -1):
                # start is found
                break

            cv2.line(map_show, (RRTtree[node_id][1], RRTtree[node_id][0]), (RRTtree[parent_id][1], RRTtree[parent_id][0]), (0, 0, 255), 1)
            cv2.circle(map_show, (RRTtree[parent_id][1], RRTtree[parent_id][0]), 2, (0,0,255), -1)
            cv2.imshow('map', map_show)
            path.append([RRTtree[parent_id][0], RRTtree[parent_id][1]])

            node_id = parent_id

    return path_found, np.array(path)

            
    
if __name__ == "__main__":

    # global map, map_show
    global target_x, target_y

    points_path = "./semantic_3d_pointcloud/point.npy"
    colors_path = "./semantic_3d_pointcloud/color01.npy"
    # # colors_path = "./semantic_3d_pointcloud/color0255.npy"

    pcd = load_point_cloud(points_path, colors_path)
    pcd = remove_ceiling_and_floor(pcd)
    # plot_scatter_graph(pcd)
    
    color_code, color_set, color_label = read_color_code()

    map_path = './map.png'
    map = cv2.imread(map_path)
    map_show = cv2.imread(map_path)


    find_pixel_voxel_scale(map, color_set)


    targets = find_target(color_code, map)
    target_name = input("please enter the target name (refrigerator, rack, cushion, lamp, cooktop): ")
    target_y = targets[target_name][0]
    target_x = targets[target_name][1]
    target_label = color_label[target_name]

    cv2.imshow('map', map_show)
    
    
    # setting mouse handler for the image
    # and calling the click_event() function
    cv2.setMouseCallback('map', click_event)

    # wait for a key to be pressed to exit
    cv2.waitKey(0)
 
    # close the window
    cv2.destroyAllWindows()

    find_target(color_code, map)

    



    path_found, path = RRT()

    
    cv2.imwrite('./output_img/' + target_name + '/path.png', map_show)

    # wait for a key to be pressed to exit
    cv2.waitKey(0)
 
    # close the window
    cv2.destroyAllWindows()


    ##################################################################
    # This is the scene we are going to load.
    # support a variety of mesh formats, such as .glb, .gltf, .obj, .ply
    ### put your scene path ###
    test_scene = "./replica_v1/apartment_0/habitat/mesh_semantic.ply"
    ######

    sim_settings = {
        "scene": test_scene,  # Scene path
        "default_agent": 0,  # Index of the default agent
        "sensor_height": 1.5,  # Height of sensors in meters, relative to the agent
        "width": 512,  # Spatial resolution of the observations
        "height": 512,
        "sensor_pitch": 0,  # sensor pitch (x rotation in rads)
    }


    ######

    ######

    img_path = './output_img/' + target_name + '/img'
    video_path = './output_video/' + target_name + '/'
    FORWARD_KEY="w"
    LEFT_KEY="a"
    RIGHT_KEY="d"
    FINISH="f"
    print("#############################")
    print("use keyboard to control the agent")
    print(" w for go forward  ")
    print(" a for turn left  ")
    print(" d for trun right  ")
    print(" f for finish and quit the program")
    print("#############################")
    
    path_voxel = np.zeros(path.shape)
    for i in range(path.shape[0]):
        path_voxel[i, 0], path_voxel[i, 1] = pixel_to_voxel(path[i, :])
    
    action_id = 0
    img_list = []
    for i in range(path_voxel.shape[0]-1, -1, -1):
        print(path_voxel[i, :])

        if(i==path_voxel.shape[0]-1):
        
            # starting point
            cfg = make_simple_cfg(sim_settings)
            sim = habitat_sim.Simulator(cfg)


            # initialize an agent
            agent = sim.initialize_agent(sim_settings["default_agent"])

            # Set agent state
            agent_state = habitat_sim.AgentState()
            agent_state.position = np.array([path_voxel[i, 0], 0.0, path_voxel[i, 1]])  # agent in world space
            agent.set_state(agent_state)

            # obtain the default, discrete actions that an agent can perform
            # default action space contains 3 actions: move_forward, turn_left, and turn_right
            action_names = list(cfg.agents[sim_settings["default_agent"]].action_space.keys())
            print("Discrete action space: ", action_names)

            # camera initially looks along -z axis
            pre_forward = np.array([0.0, -1.0])
            pre_length = 1


            continue
        
        keystroke = cv2.waitKey(0)

        if keystroke == ord(FINISH):
            print("action: FINISH")
            break

        forward = path_voxel[i, :] - path_voxel[i+1, :]
        length = np.linalg.norm(forward)

        
        if(length!=0 and pre_length!=0):

            if(np.dot(pre_forward, forward)/length/pre_length <= 1):
            
                # dot -> to know how many degree to turn
                print((forward != pre_forward), (forward != pre_forward).any())
                print(pre_forward,forward)
                theta = np.rad2deg(np.arccos(np.dot(pre_forward, forward)/length/pre_length))

                # cross -> to know turn left or turn right
                if(np.cross(pre_forward, forward) > 0):
                    action = "turn_right"
                else: 
                    action = "turn_left"

                img = navigateAndSee(target_label, action_names, agent, sim, action, theta)
                cv2.imshow("RGB_MASK", img)
                cv2.imwrite(img_path + str(action_id) + '.png', img)
                img_list.append(img)
                action_id = action_id + 1

                keystroke = cv2.waitKey(0)
                if keystroke == ord(FINISH):
                    print("action: FINISH")
                    break

        action = "move_forward"
        img = navigateAndSee(target_label, action_names, agent, sim, action, length)
        cv2.imshow("RGB_MASK", img)
        cv2.imwrite(img_path + str(action_id) + '.png', img)
        img_list.append(img)
        action_id = action_id + 1

        pre_forward = forward
        pre_length = length
        
    
    if(path_found):
        video_path = video_path + target_name + '.mp4'
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        size = (img_list[0].shape[0:2])
        #for img_name in glob.glob('./output_img/' + target_name + '/*.png'):
        #    img = cv2.imread(img_name)
        #    height, width, layer = img.shape
        #    size = (width, height)
        #    img_list.append(img)
            
        # print(video_path, size)
        out = cv2.VideoWriter(video_path, fourcc, 1.0, size)
        for img in img_list:

            out.write(img)
        out.release()
    
        
