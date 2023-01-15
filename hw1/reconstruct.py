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



def preprocess_point_cloud(pcd, voxel_size):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

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

def find_nearest_neighbor(source, target):
    # source.shape = (n,3), target.shape = (m,3)
    # print(source.shape, target.shape)
    id = np.zeros((source.shape[0]))             # shape = (n, ), to store the nearest neighbor id in target for points in source
    dis = np.ones((source.shape[0]))*np.inf      # shape = (n, ), to store the nearest neighbor distance in target for points in source (initialize to infinite)
    correspond_target = np.zeros(source.shape)   # shape = (n, 3)
   
    for i in range(source.shape[0]):
        for j in range(target.shape[0]):

            if(np.linalg.norm(source[i]-target[j]) < dis[i]):
                id[i] = int(j)
                dis[i] = np.linalg.norm(source[i]-target[j])

        # print(id[i])
        correspond_target[i, :] = target[int(id[i]), :]
    
    
    return correspond_target.transpose()

def find_best_transform(source, target):
    # source.shape = target.shape = (3, n)
    source_mean = np.mean(source, axis=1)   # shape = (3, )
    target_mean = np.mean(target, axis=1)   # shape = (3, )
    source_prime = source - np.tile(source_mean,(source.shape[1],1)).transpose()
    target_prime = target - np.tile(target_mean,(target.shape[1],1)).transpose()

    u, sigma, vt = np.linalg.svd(np.dot(source_prime, target_prime.transpose()))

    rotation = np.dot(vt.transpose(), u.transpose())
    translate = target_mean - np.dot(rotation, source_mean)
    transform_mat = np.zeros((4, 4))        # homogeneous
    transform_mat[3, 3] = 1
    transform_mat[0:3, 0:3] = rotation
    transform_mat[0:3, 3] = translate

    return transform_mat
                


def refine_registration_by_me(source, target, initial_transform_mat, distance_threshold):
    # distance_threshold = voxel_size * 0.4

    # return initial_transform_mat

    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)

    source_points = np.asarray(source.points)       # shape = (n, 3)
    target_points = np.asarray(target.points)       # shape = (m, 3)
    # print(source_points, source_points.shape)
    # print(target_points, target_points.shape)

    # homogeneous
    source = np.ones((4, source_points.shape[0]))   # shape = (4, n)
    target = np.ones((4, target_points.shape[0]))   # shape = (4, m)
    source[0:3, :] = source_points.transpose()
    target[0:3, :] = target_points.transpose()

    # apply initial_transform_mat to source
    source = np.dot(initial_transform_mat, source)

    # error = 0
    max_iterate = 50
    result_transform_mat = initial_transform_mat

    for i in range(max_iterate):
        
        correspond_target = find_nearest_neighbor(source.transpose()[:, 0:3], target_points)    # shape = (3, n)
        transform_mat = find_best_transform(source[0:3, :], correspond_target)                  # homogeneous
        # print(transform_mat)

        # update current source
        source = np.dot(transform_mat, source)

        error = 0
        for j in range(source.shape[1]):
            error = error + np.linalg.norm(source[0:3, j]-correspond_target[:, j])

        result_transform_mat = np.dot(transform_mat, result_transform_mat)
        if(error < distance_threshold):
            break
    
    return result_transform_mat

def create_trajectory_line(data_num):
    lines = np.zeros((data_num-1, 2), dtype=int)
    for i in range(data_num-1):
        lines[i, :] = [i, i+1]
    return lines

if __name__ == "__main__":

    # read file to get ground truth
    data_num = 159
    ground_truth_points = np.zeros((data_num, 3))
    ground_truth_lines = create_trajectory_line(data_num)
    row_id = 0
    with open('./task2_data/camera_pose.csv', newline='') as csvfile:
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
    # ground_truth_line_set.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

    # camera intrinsic mat
    resolution = 512
    fov = math.pi/2
    focal_length = float(resolution/2*mpmath.cot(fov/2)) 
    # intrinsic = o3d.camera.PinholeCameraIntrinsic(resolution, resolution, focal_length, focal_length, resolution/2, resolution/2)
    intrinsic = np.array([[focal_length, 0, resolution/2], [0, focal_length, resolution/2], [0, 0, 1]])

    # color1 = o3d.io.read_image("./task2_data/color/color1.png")
    

    pcds = []
    pcds2 = []
    pcds_down = []
    rgbd_imgs = []
    fpfhs = []
    fpfhs2 = []
    voxel_size = 0.1    # icp by o3d
    # voxel_size = 0.0000001
    voxel_size2 = 0.4   # icp by me -> floor 1
    # voxel_size2 = 0.4   # icp by me -> floor 2
    depth_scale = 1000
    transform_mats_by_me = []
    transform_mats_by_o3d = []
    transform_mats_by_me.append(np.identity(4))
    transform_mats_by_o3d.append(np.identity(4))
    

    for i in range(data_num):
        print(i)

        # color_name = "./task2_data/color/color" + str(i) + ".png"
        # depth_name = "./task2_data/depth/depth" + str(i) + ".png"

        color_name = "./task2_data/color/color" + str(i+1) + ".png"
        depth_name = "./task2_data/depth/depth" + str(i+1) + ".png"

        # rgb_img = cv2.imread(color_name)    # bgr
        # rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        # rgb_img = o3d.geometry.Image(rgb_img)
        # depth_img = o3d.io.read_image(depth_name)
        # depth_img = np.asarray(depth_img, dtype=np.float32)
        # depth_img = depth_img/255*10*depth_scale
        # depth_img = o3d.geometry.Image(depth_img)


        rgb_img = cv2.imread(color_name)    # bgr
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        depth_img = o3d.io.read_image(depth_name)
        depth_img = np.asarray(depth_img, dtype=np.float32)
        depth_img = depth_img/255*10


        # rgbd_img, pcd = depth_image_to_point_cloud(rgb_img, depth_img, intrinsic)
        rgbd_img, pcd = depth_image_to_point_cloud_by_me(rgb_img, depth_img, intrinsic)
        pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:, 1] < 0.5)[0])   # remove ceiling -> floor 1
        # pcd = pcd.select_by_index(np.where(np.asarray(pcd.points)[:, 1] < 0.5)[0])   # remove ceiling -> floor 2
        
        
        pcd_down, fpfh = preprocess_point_cloud(pcd, voxel_size)

        
        # print(np.asarray(pcd2.points).shape, np.asarray(pcd.points).shape)
        pcd2, ind = pcd.remove_statistical_outlier(nb_neighbors=20,
                                                     std_ratio=2)
                                                
        pcd2, fpfh2 = preprocess_point_cloud(pcd2, voxel_size2)
        # pcd2, ind = pcd2.remove_radius_outlier(nb_points=16, radius=0.6) 
        
        
        # print(np.asarray(pcd.points))
        pcds.append(pcd)
        pcds2.append(pcd2)
        rgbd_imgs.append(rgbd_img)
        pcds_down.append(pcd_down)
        fpfhs.append(fpfh)
        fpfhs2.append(fpfh2)

        

        if(i!=0):
    
            source = pcds[i]
            target = pcds[i-1]
            source_down = pcds_down[i]
            target_down = pcds_down[i-1]
            source_down2 = pcds2[i]
            target_down2 = pcds2[i-1]
            source_fpfh = fpfhs[i]
            target_fpfh = fpfhs[i-1]
            source_fpfh2 = fpfhs2[i]
            target_fpfh2 = fpfhs2[i-1]


            

            # icp by o3d
            result_ransac = execute_global_registration(source_down, target_down,
                                            source_fpfh, target_fpfh,
                                            voxel_size)

            result_icp_by_o3d = refine_registration(source_down, target_down, source_fpfh, target_fpfh,
                                voxel_size, result_ransac.transformation)
            transform_mats_by_o3d.append(np.dot(transform_mats_by_o3d[i-1], result_icp_by_o3d.transformation))


            # icp by me
            result_ransac2 = execute_global_registration(source_down2, target_down2,
                                            source_fpfh2, target_fpfh2,
                                            voxel_size2)

            result_icp_by_me = refine_registration_by_me(source_down2, target_down2, result_ransac2.transformation, voxel_size2*0.4)
            
            transform_mats_by_me.append(np.dot(transform_mats_by_me[i-1], result_icp_by_me))



    # icp by o3d        
    pcd_combined1 = pcds[0]
    pcd_combined2 = pcds[0]
    for i in range(1, len(pcds)):
        print('combine: '+str(i))
        pcd_combined1 = pcd_combined1 + copy.deepcopy(pcds[i]).transform(transform_mats_by_o3d[i])
        pcd_combined2 = pcd_combined2 + copy.deepcopy(pcds[i]).transform(transform_mats_by_me[i])
    # pcd_combined_down1 = pcd_combined1.voxel_down_sample(voxel_size=0.00000005)
    # pcd_combined_down2 = pcd_combined2.voxel_down_sample(voxel_size=0.00000005)
    # print(np.max(np.asarray(pcd_combined1.points)[:,1]))

   
    colors = [[1, 0, 0] for i in range(len(ground_truth_lines))]
    icp_by_o3d_points = np.zeros((data_num, 3))
    icp_by_me_points = np.zeros((data_num, 3))
    
    for i in range(data_num):
        icp_by_o3d_points[i, :] = transform_mats_by_o3d[i][0:3,3]
        icp_by_me_points[i, :] = transform_mats_by_me[i][0:3,3]


    
    icp_by_o3d_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(icp_by_o3d_points),
        lines=o3d.utility.Vector2iVector(ground_truth_lines),
    )
    icp_by_o3d_line_set.colors = o3d.utility.Vector3dVector(colors)

    icp_by_me_line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(icp_by_me_points),
        lines=o3d.utility.Vector2iVector(ground_truth_lines),
    )
    icp_by_me_line_set.colors = o3d.utility.Vector3dVector(colors)


    icp_by_o3d_L2_dis = np.mean(np.linalg.norm(icp_by_o3d_points - ground_truth_points, axis=1))
    icp_by_me_L2_dis = np.mean(np.linalg.norm(icp_by_me_points - ground_truth_points, axis=1))

    print('L2 distance: ')
    print('ICP by o3d: ', icp_by_o3d_L2_dis)
    print('ICP by me: ', icp_by_me_L2_dis)

    
    o3d.visualization.draw_geometries([pcd_combined1, icp_by_o3d_line_set, ground_truth_line_set])
    o3d.visualization.draw_geometries([pcd_combined2, icp_by_me_line_set, ground_truth_line_set])


    