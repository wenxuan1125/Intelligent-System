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
from collections import Counter
import threading

points_voxel_dict = {}
sem = threading.Semaphore(8)

def depth_image_to_point_cloud_by_me(rgb_img, depth_img, intrinsic):

    # rgb_img.shape = (512, 512, 3)
    # depth_img.shape = (512, 512)
    

    point_num = rgb_img.shape[0] * rgb_img.shape[0]
    points_xyz = np.zeros((point_num, 3))
    points_color = np.zeros((point_num, 3))
    rgbd_img = np.zeros((rgb_img.shape[0], rgb_img.shape[0], 4))
    rgbd_img[:, :, 0:3] = rgb_img
    rgbd_img[:, :, 3] = depth_img
    id = 0

    for u in range(rgb_img.shape[0]):
        for v in range(rgb_img.shape[1]):

            
            uv = np.array([v, u, 1]).transpose()                        # homogeneous
            xyz = np.dot(scipy.linalg.inv(intrinsic), uv)               # image plane to 3D coordinate
            xyz = xyz * depth_img[u, v]
            points_xyz[id, :] = xyz
            points_color[id, :] = rgb_img[u, v, :]/255

            
            id = id +1

    # print(np.max(points_xyz[:, 1]))
    # print(points_xyz)

    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz)
    pcd.colors = o3d.utility.Vector3dVector(points_color)
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    return rgbd_img, pcd



def depth_image_to_point_cloud(rgb_img, depth_img, intrinsic):
    # intrinsic_mat: 
    # intrinsic matrix transforms 3D coordinates to 2D coordinates on an image plane 
    # using the pinhole camera model.

    # Pinhole camera model
    width = 512             # Spatial resolution
    height = 512
    fov = 90                # degree
    depth_scale = 1000
    

    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_img, depth_img, convert_rgb_to_intensity=False); 

    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic)
    # Flip it, otherwise the pointcloud will be upside down
    pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    # o3d.visualization.draw_geometries([pcd])

    return rgbd_image, pcd

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])

def custom_voxel_down(pcd, voxel_size):
    voxel_size3d = np.array([voxel_size, voxel_size, voxel_size])
    pcd_points = np.asarray(pcd.points)         # shape = (n, 3)
    pcd_colors = np.asarray(pcd.colors)
    voxel_index = np.zeros(pcd_points.shape)    # shape = (n, 3)

    pcd_points_min = pcd_points.min(axis=0)
    pcd_points_max = pcd_points.max(axis=0)
    pcd_points_min_bound = pcd_points_min - voxel_size3d * 0.5
    pcd_points_max_bound = pcd_points_max + voxel_size3d * 0.5
    points_in_voxel = {}

    

    for i in range(pcd_points.shape[0]):
        voxel_index[i, :] = np.floor((pcd_points[i, :] - pcd_points_min_bound)/ voxel_size3d)

        # np.array is not hashable, cannot be the key of the dictionary
        # tuple is hashable
        # np.array to tuple
        if tuple(voxel_index[i, :]) in points_in_voxel:
            points_in_voxel[tuple(voxel_index[i, :])].append(i)
        else:
            points_in_voxel[tuple(voxel_index[i, :])] = [i]

    pcd_down_points = np.zeros((len(points_in_voxel), 3))    # shape = (n, 3) 
    pcd_down_colors = np.zeros((len(points_in_voxel), 3))    # shape = (n, 3) 
    i = 0
    for voxel_id, points in points_in_voxel.items():
        # print(i, len(points_in_voxel), voxel_id, pcd_points[points].shape)
        # print(i, len(points_in_voxel))
        if (pcd_points[points].shape[0] == 1):
            pcd_down_points[i, :] = pcd_points[points][0]
        else:
            pcd_down_points[i, :] = pcd_points[points].mean(axis=0)

        colors = pcd_colors[points].tolist()
        colors = [tuple(c) for c in colors]
        
        rgb = Counter(colors).most_common()[0][0]
        pcd_down_colors[i, :] = rgb

        # r = Counter(pcd_colors[points][:, 0]).most_common()[0][0]
        # g = Counter(pcd_colors[points][:, 1]).most_common()[0][0]
        # b = Counter(pcd_colors[points][:, 2]).most_common()[0][0]
        # pcd_down_colors[i, :] = np.array([r, g, b])
        i = i + 1

    pcd_down = o3d.geometry.PointCloud()
    pcd_down.points = o3d.utility.Vector3dVector(pcd_down_points)
    pcd_down.colors = o3d.utility.Vector3dVector(pcd_down_colors)

    return pcd_down

# def thread_job1(i, pcd_points, pcd_points_min_bound, voxel_size3d):
#     voxel_index = np.floor((pcd_points - pcd_points_min_bound)/ voxel_size3d)

#     # np.array is not hashable, cannot be the key of the dictionary
#     # tuple is hashable
#     # np.array to tuple
#     if tuple(voxel_index) in points_voxel_dict:
#         points_voxel_dict[tuple(voxel_index)].append(i)
#     else:
#         points_voxel_dict[tuple(voxel_index)] = [i]

#     sem.release()

# def thread_job2(i, points):
#     if (pcd_points[points].shape[0] == 1):
#         pcd_down_points[i, :] = pcd_points[points][0]
#     else:
#         pcd_down_points[i, :] = pcd_points[points].mean(axis=0)


#     r = Counter(pcd_colors[points][:, 0]).most_common()[0][0]
#     g = Counter(pcd_colors[points][:, 1]).most_common()[0][0]
#     b = Counter(pcd_colors[points][:, 2]).most_common()[0][0]
#     pcd_down_colors[i, :] = np.array([r, g, b])


#     sem.release()
    
# def custom_voxel_down_multithread(pcd, voxel_size):
#     voxel_size3d = np.array([voxel_size, voxel_size, voxel_size])
#     global pcd_points, pcd_colors
#     pcd_points = np.asarray(pcd.points)         # shape = (n, 3)
#     pcd_colors = np.asarray(pcd.colors)
#     voxel_index = np.zeros(pcd_points.shape)    # shape = (n, 3)

#     pcd_points_min = pcd_points.min(axis=0)
#     pcd_points_max = pcd_points.max(axis=0)
#     pcd_points_min_bound = pcd_points_min - voxel_size3d * 0.5
#     pcd_points_max_bound = pcd_points_max + voxel_size3d * 0.5
    

    

#     for i in range(pcd_points.shape[0]):
#         sem.acquire()
#         threading.Thread(target=thread_job1, args=(i, pcd_points[i, :], pcd_points_min_bound, voxel_size3d)).start()

#     # wait for all threads completing
#     for i in range(8):
#         sem.acquire()
#     for i in range(8):
#         sem.release()


#     global pcd_down_points, pcd_down_colors
#     pcd_down_points = np.zeros((len(points_voxel_dict), 3))    # shape = (n, 3) 
#     pcd_down_colors = np.zeros((len(points_voxel_dict), 3))    # shape = (n, 3) 
 


#     i = 0
#     for voxel_id, points in points_voxel_dict.items():
#         # print(i, len(points_in_voxel), voxel_id, pcd_points[points].shape)
#         print(i, len(points_voxel_dict))

#         sem.acquire()
#         threading.Thread(target=thread_job2, args=(i, points)).start()


#         # r = Counter(pcd_colors[:, 0]).most_common()[0][0]
#         # g = Counter(pcd_colors[:, 1]).most_common()[0][0]
#         # b = Counter(pcd_colors[:, 2]).most_common()[0][0]
#         # pcd_down_colors[i, :] = np.array([r, g, b])
#         i = i + 1

#     # wait for all threads completing
#     for i in range(8):
#         sem.acquire()
#     for i in range(8):
#         sem.release()

#     pcd_down = o3d.geometry.PointCloud()
#     pcd_down.points = o3d.utility.Vector3dVector(pcd_down_points)
#     pcd_down.colors = o3d.utility.Vector3dVector(pcd_down_colors)

#     return pcd_down


def preprocess_point_cloud(pcd, voxel_size):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    # pcd_down = pcd.voxel_down_sample(voxel_size)
    # pcd_down = custom_voxel_down_multithread(pcd, voxel_size)
    pcd_down = custom_voxel_down(pcd, voxel_size)

    # o3d.visualization.draw_geometries([pcd_down])

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))

    return pcd_down, pcd_fpfh

def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    return result

# open3d icp
def refine_registration(source, target, source_fpfh, target_fpfh, voxel_size, initial_transform_mat):
    distance_threshold = voxel_size * 0.4
    # print(":: Point-to-plane ICP registration is applied on original point")
    # print("   clouds to refine the alignment. This time we use a strict")
    # print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, initial_transform_mat,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result


def create_trajectory_line(data_num):
    lines = np.zeros((data_num-1, 2), dtype=int)
    for i in range(data_num-1):
        lines[i, :] = [i, i+1]
    return lines
def read_pose_ground_truth():
    ground_truth_points = np.zeros((data_num, 3))
    ground_truth_lines = create_trajectory_line(data_num)
    row_id = 0
    with open('./reconstruct/floor2/camera_pose.csv', newline='') as csvfile:
        rows = csv.reader(csvfile)
        
        for row in rows:
            if(row_id >= data_num):
                break
            x = float(row[0])
            y = float(row[1])
            z = float(row[2])

            if(row_id==0):
                origin = np.array([x, y, z])
            
            ground_truth_points[row_id, :] = np.array([x, y, z]) - origin
            row_id = row_id + 1

    ground_truth_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(ground_truth_points),
        lines=o3d.utility.Vector2iVector(ground_truth_lines),
    )

    return ground_truth_points, ground_truth_lines, ground_truth_line_set


if __name__ == "__main__":

    # read file to get ground truth
    data_num = 155
    ground_truth_points, ground_truth_lines, ground_truth_line_set = read_pose_ground_truth()

    # camera intrinsic mat
    resolution = 512
    fov = math.pi/2
    focal_length = float(resolution/2*mpmath.cot(fov/2)) 
    intrinsic = o3d.camera.PinholeCameraIntrinsic(resolution, resolution, focal_length, focal_length, resolution/2, resolution/2)
    # intrinsic = np.array([[focal_length, 0, resolution/2], [0, focal_length, resolution/2], [0, 0, 1]])


    

    pcds = []
    pcds_down = []
    rgbd_imgs = []
    fpfhs = []
    voxel_size = 0.15    # icp by o3d
    # voxel_size = 0.3
    depth_scale = 1000
    transform_mats_by_o3d = []
    transform_mats_by_o3d.append(np.identity(4))
    
    
    for i in range(data_num):
        print(i)

        points_voxel_dict = {}


        # color_name = "./reconstruct/floor2/predict/apartment0/predict" + str(i+1) + ".png"
        color_name = "./reconstruct/floor2/ground/ground" + str(i+1) + ".png"
        depth_name = "./reconstruct/floor2/depth/depth" + str(i+1) + ".png"


        rgb_img = cv2.imread(color_name)    # bgr
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        rgb_img = o3d.geometry.Image(rgb_img)
        depth_img = o3d.io.read_image(depth_name)
        depth_img = np.asarray(depth_img, dtype=np.float32)
        depth_img = depth_img/255*10*depth_scale
        depth_img = o3d.geometry.Image(depth_img)



      

        rgbd_img, pcd = depth_image_to_point_cloud(rgb_img, depth_img, intrinsic)
        # rgbd_img, pcd = depth_image_to_point_cloud_by_me(rgb_img, depth_img, intrinsic)
        pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:, 1] < 0.5)[0])   # remove ceiling -> floor 1
        # pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:, 1] < 0.5)[0])   # remove ceiling -> floor 2
        
        
        pcd_down, fpfh = preprocess_point_cloud(pcd, voxel_size)


        
        pcds.append(pcd)
        rgbd_imgs.append(rgbd_img)
        pcds_down.append(pcd_down)
        fpfhs.append(fpfh)
        

        if(i!=0):
    
            source = pcds[i]
            target = pcds[i-1]
            source_down = pcds_down[i]
            target_down = pcds_down[i-1]
            source_fpfh = fpfhs[i]
            target_fpfh = fpfhs[i-1]
    

            

            # icp by o3d
            result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)

            result_icp_by_o3d = refine_registration(source_down, target_down, source_fpfh, target_fpfh,
                                voxel_size, result_ransac.transformation)
            transform_mats_by_o3d.append(np.dot(transform_mats_by_o3d[i-1], result_icp_by_o3d.transformation))





    # icp by o3d        
    pcd_combined = pcds[0]
    for i in range(1, len(pcds)):
        print('combine: '+str(i))
        pcd_combined = pcd_combined + copy.deepcopy(pcds[i]).transform(transform_mats_by_o3d[i])
    # print(np.max(np.asarray(pcd_combined.points)[:,1]))
    # pcd_combined = custom_voxel_down(pcd_combined, 0.01)
   
    colors = [[1, 0, 0] for i in range(len(ground_truth_lines))]
    icp_by_o3d_points = np.zeros((data_num, 3))

    
    for i in range(data_num):
        icp_by_o3d_points[i, :] = transform_mats_by_o3d[i][0:3,3]

    
    icp_by_o3d_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(icp_by_o3d_points),
        lines=o3d.utility.Vector2iVector(ground_truth_lines),
    )
    icp_by_o3d_line_set.colors = o3d.utility.Vector3dVector(colors)



    icp_by_o3d_L2_dis = np.mean(np.linalg.norm(icp_by_o3d_points - ground_truth_points, axis=1))

    print('L2 distance: ')
    print('ICP by o3d: ', icp_by_o3d_L2_dis)

    
    o3d.visualization.draw_geometries([pcd_combined, icp_by_o3d_line_set, ground_truth_line_set])


    