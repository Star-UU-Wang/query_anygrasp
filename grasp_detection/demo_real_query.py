import os
import argparse
import torch
import numpy as np
import open3d as o3d
from PIL import Image

import json

from gsnet import AnyGrasp
from graspnetAPI import GraspGroup

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps.')
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))

def demo(data_dir):
    anygrasp = AnyGrasp(cfgs)
    anygrasp.load_net()

    # os.getcwd()
    record_timestamps_path = os.path.join(data_dir, "out", "record_timestamps.txt")
    # read timestamps
    with open(record_timestamps_path, "r") as file:
        timestamps = file.read().splitlines()

    # get latest timestamp
    if timestamps:
        latest_timestamp = timestamps[-1]
    else:
        print("No timestamps found in the file.")

    OUT_PATH = os.path.join(data_dir, "out", latest_timestamp)



    fx, fy = 924.034423828125, 924.2615966796875
    cx, cy = 656.5450439453125, 360.77032470703125
    # scale = 1000.0
    with open(os.path.join(OUT_PATH, 'depth_scale.json'), 'r') as f:
        data = json.load(f)
    scale = np.array(data['scale'])

    # K: [924.034423828125, 0.0, 656.5450439453125, 0.0, 924.2615966796875, 360.77032470703125, 0.0, 0.0, 1.0]
    # set workspace to filter output grasps
    xmin, xmax = -0.19, 0.12
    ymin, ymax = 0.02, 0.15
    zmin, zmax = 0.0, 1.0
    lims = [xmin, xmax, ymin, ymax, zmin, zmax]

    # get raw point
    raw_colors = np.array(Image.open(os.path.join(OUT_PATH, 'color.png')), dtype=np.float32) / 255.0
    raw_depths = np.array(Image.open(os.path.join(OUT_PATH, 'depth.png')))

    # get point cloud
    xmap, ymap = np.arange(raw_depths.shape[1]), np.arange(raw_depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = raw_depths * scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    mask = (points_z > 0) & (points_z < 1)
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask].astype(np.float32)
    raw_colors = raw_colors[mask].astype(np.float32)
    print(points.min(axis=0), points.max(axis=0))

    raw_gg, raw_cloud = anygrasp.get_grasp(points, raw_colors, lims=lims, apply_object_mask=True, dense_grasp=False, collision_detection=True)

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(raw_colors)
    o3d.io.write_point_cloud(os.path.join(OUT_PATH, "raw_cloud.ply"), cloud)
    print(f"Point cloud saved to {os.path.join(OUT_PATH, 'raw_cloud.ply')}")

# ============================================================================================================== #



    # get data
    colors = np.array(Image.open(os.path.join(OUT_PATH, 'light_hqsam_mask_color.png')), dtype=np.float32) / 255.0
    depths = np.array(Image.open(os.path.join(OUT_PATH, 'light_hqsam_mask_depth.png')))
    # colors = np.array(Image.open(os.path.join(OUT_PATH, 'mask_color.png')), dtype=np.float32) / 255.0
    # depths = np.array(Image.open(os.path.join(OUT_PATH, 'mask_depth.png')))
    # fx, fy = 927.17, 927.37
    # cx, cy = 651.32, 349.62
    # scale = 1000.0
    
    # get camera intrinsics -> realsense d435i mounted on xarm

    # get point cloud
    xmap, ymap = np.arange(depths.shape[1]), np.arange(depths.shape[0])
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depths * scale
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    # set your workspace to crop point cloud
    mask = (points_z > 0) & (points_z < 1)
    points = np.stack([points_x, points_y, points_z], axis=-1)
    points = points[mask].astype(np.float32)
    colors = colors[mask].astype(np.float32)
    print(points.min(axis=0), points.max(axis=0))

    gg, cloud = anygrasp.get_grasp(points, colors, lims=lims, apply_object_mask=True, dense_grasp=False, collision_detection=True)

    if len(gg) == 0:
        print('No Grasp detected after collision detection!')

    gg = gg.nms().sort_by_score()
    gg_pick = gg[0:20]
    print(gg_pick.scores)
    print('grasp score:', gg_pick[0].score)

    # seve point cloud
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    o3d.io.write_point_cloud(os.path.join(OUT_PATH, "cloud.ply"), cloud)
    print(f"Point cloud saved to {os.path.join(OUT_PATH, 'cloud.ply')}")

    gripper_poses = {}
    for i, gripper in enumerate(gg):
        translation = gripper.translation
        rotation = gripper.rotation_matrix
        RT = np.eye(4)
        RT[:3, 3] = translation
        RT[:3, :3] = rotation
        gripper_poses[f"Gripper{i}"] = RT.tolist()

    with open(os.path.join(OUT_PATH, "gripper_6d_poses.json"), "w") as f:
        json.dump(gripper_poses, f, indent=4)
    print(f"Gripper 6D poses saved to {os.path.join(OUT_PATH, 'gripper_6d_poses.json')}")

    # gripper[0]
    with open(os.path.join(OUT_PATH, "gripper_pose.json"), "w") as f:
        json.dump({'pose':gripper_poses["Gripper0"]}, f, indent=4)

    # visualization
    if cfgs.debug:
        # trans_mat = np.array([[1,0,0,0],[0,1,0,0],[0,0,-1,0],[0,0,0,1]])
        # cloud.transform(trans_mat)
        coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1, origin=[0, 0, 0])

        grippers = gg.to_open3d_geometry_list()
        # for gripper in grippers:
        #     gripper.transform(trans_mat)
        o3d.visualization.draw_geometries([*grippers, raw_cloud, coord_frame])
        o3d.visualization.draw_geometries([grippers[0], raw_cloud, coord_frame])


if __name__ == '__main__':
    
    # demo('./example_data/')
    par_path = os.getcwd()
    demo(par_path)