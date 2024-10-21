import numpy as np
import math
import json

# def euler_to_rotation_matrix(roll, pitch, yaw):
#     """
#     Convert Euler angles (roll, pitch, yaw) to a rotation matrix.
    
#     Parameters:
#         roll:   Rotation around the x-axis (in radians)
#         pitch:  Rotation around the y-axis (in radians)
#         yaw:    Rotation around the z-axis (in radians)
        
#     Returns:
#         R:      3x3 rotation matrix
#     """
#     # Roll, pitch, and yaw angles
#     cos_roll = math.cos(roll)
#     sin_roll = math.sin(roll)
#     cos_pitch = math.cos(pitch)
#     sin_pitch = math.sin(pitch)
#     cos_yaw = math.cos(yaw)
#     sin_yaw = math.sin(yaw)

#     # Compute rotation matrix
#     R_roll = np.array([[1, 0, 0],
#                        [0, cos_roll, -sin_roll],
#                        [0, sin_roll, cos_roll]])
    
#     R_pitch = np.array([[cos_pitch, 0, sin_pitch],
#                         [0, 1, 0],
#                         [-sin_pitch, 0, cos_pitch]])
    
#     R_yaw = np.array([[cos_yaw, -sin_yaw, 0],
#                       [sin_yaw, cos_yaw, 0],
#                       [0, 0, 1]])

#     # Combine the rotations
#     R = np.dot(R_roll, np.dot(R_pitch, R_yaw))

#     return R

def eulerAngles2rotationMat(theta, format='degree'):
    """
    Calculates Rotation Matrix given euler angles.
    :param theta: 1-by-3 list [rx, ry, rz] angle in degree
    :return:
    """
    if format == 'degree':
        theta = [i * math.pi / 180.0 for i in theta]
 
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(theta[0]), -math.sin(theta[0])],
                    [0, math.sin(theta[0]), math.cos(theta[0])]
                    ])
 
    R_y = np.array([[math.cos(theta[1]), 0, math.sin(theta[1])],
                    [0, 1, 0],
                    [-math.sin(theta[1]), 0, math.cos(theta[1])]
                    ])
 
    R_z = np.array([[math.cos(theta[2]), -math.sin(theta[2]), 0],
                    [math.sin(theta[2]), math.cos(theta[2]), 0],
                    [0, 0, 1]
                    ])
    R = np.dot(R_z, np.dot(R_y, R_x))
    return R

if __name__ == '__main__':

    RT_h2b = np.array([[0.] * 4 for _ in range(4)])
    rotation_matrix = eulerAngles2rotationMat([177.4, -3.3, 0.7], format='degree')
    RT_h2b[:3, :3] = rotation_matrix
    RT_h2b[:3, 3] = np.array([0.3067, 0.0058, 0.7106])
    RT_h2b[3, 3] = 1

    RT_c2h = np.array([[0.] * 4 for _ in range(4)])
    # RT_c2h[:3, :3] = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    RT_c2h[:3, :3] = eulerAngles2rotationMat([0, 0, 90], format='degree')
    RT_c2h[:3, 3] = np.array([0.0675, -0.0175, 0.025375])
    RT_c2h[3, 3] = 1

    RT_c2b = RT_h2b @ RT_c2h

    # Specify the filename
    filename = "c2b-pose-new.json"

# Save the list representation of the array to a JSON file
    with open(filename, 'w') as f:
        json.dump({'pose':RT_c2b.tolist()}, f, indent=4)


# x +67.5mm
# y -17.5mm
# z +25.375mm
