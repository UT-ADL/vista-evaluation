from transformations import quaternion_from_euler
  
import numpy as np # Scientific computing library for Python
 
def get_quaternion_from_euler(roll, pitch, yaw):
  """
  Convert an Euler angle to a quaternion.
   
  Input
    :param roll: The roll (rotation around x-axis) angle in radians.
    :param pitch: The pitch (rotation around y-axis) angle in radians.
    :param yaw: The yaw (rotation around z-axis) angle in radians.
 
  Output
    :return qx, qy, qz, qw: The orientation in quaternion [x,y,z,w] format
  """
  qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
  qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
  qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
  qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
 
  return [qx, qy, qz, qw]


class Quaternion:
    w: float
    x: float
    y: float
    z: float

import math

def quaternion_from_euler(roll, pitch, yaw):
    """
    Converts euler roll, pitch, yaw to quaternion
    """
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)

    q = Quaternion()
    q.w = cy * cp * cr + sy * sp * sr
    q.x = cy * cp * sr - sy * sp * cr
    q.y = sy * cp * sr + cy * sp * cr
    q.z = sy * cp * cr - cy * sp * sr
    return q 

if __name__ == '__main__':

  # RPY to convert: 90deg, 0, -90deg
  #q = quaternion_from_euler(-1.602, 0.022, -1.544)
  q = get_quaternion_from_euler(-1.602, 0.022, -1.544)
  print( "The quaternion representation is %s %s %s %s." % (q[0], q[1], q[2], q[3]) )
  q = quaternion_from_euler(-1.602, 0.022, -1.544)
  print( "The quaternion representation is %s %s %s %s." % (q.x, q.y, q.z, q.w) )


  q.x = 0
  q.w = 0
  mag = math.sqrt(q.z*q.z + q.y*q.y)
  q.z /= mag;
  q.y /= mag;

  print( q.x, q.y, q.z, q.w )

from typing import Optional, Union, List, Tuple, Any
import numpy as np
from scipy.spatial.transform import Rotation

Vec = Union[np.ndarray, List[Any], Tuple[Any]]


def euler2quat(euler: Vec,
              seq: Optional[str] = 'xyz',
              degrees: Optional[bool] = False) -> Vec:
  """ Convert Euler rotation to quaternion.
  Args:
      euler (Vec): A 3-dimensional rotation vector in Euler angle.
      seq (str): The order of the rotation vector, e.g., ``xzy``;
                  default to ``xyz``.
  Returns:
      Vec: A 4-dimensional vector that describes a quaternion.
  """
  R = Rotation.from_euler(seq, euler, degrees)
  return R.as_quat()

a = [0.91965, -0.1074, -0.412]
print( euler2quat( a ) )

a2 = [-1.602, 0.022, -1.544] 
print( "this->", euler2quat( [ 0, 0, a2[2] ] ) )

a3 = [0, 0, -0.0459]
print( euler2quat( a3 ) )

print( [ 1.02+0.91965, -0.1074, 1.78734-0.3-0.412] )
# res = [0.4239435, -0.13772704, -0.15971921, 0.88079109]

# q = Quaternion()
# q.x = res[0]
# q.y = res[1]
# q.z = res[2]
# q.w = res[3]

# q.x = 0
# q.y = 0
# mag = math.sqrt(q.z*q.z + q.w*q.w)
# q.z /= mag;
# q.w /= mag;

# print( q.x, q.y, q.z, q.w )

import vista.utils.transform as tr
import numpy as np

# xyz="1.02 0 1.78734" rpy="0 0 -0.0459" 
lidar_pos = tr.vec2mat( [1.02, 0, 1.78734], [0, 0, -0.0459] ) 

# <origin xyz="0.91965 -0.1074 -0.412" rpy="-1.602 0.022 -1.544" />
camera_pos = tr.vec2mat( [0.91965, -0.1074, -0.412], [-1.602, 0.022, -1.544] ) 

from numpy.linalg import inv
#lidar_pos_inv = inv( lidar_pos )

res = np.matmul( lidar_pos, camera_pos )
#res = np.matmul( res, lidar_pos )
print( res )
pos, rpy = tr.mat2vec( res )
print( pos )
print( rpy )
print( tr.euler2quat( rpy ) )

# res = tr.euler2quat( rpy )
# q = Quaternion()
# q.x = res[0]
# q.y = res[1]
# q.z = res[2]
# q.w = res[3]

# q.x = 0
# q.z = 0
# mag = math.sqrt(q.y*q.y + q.w*q.w)
# q.y /= mag;
# q.w /= mag;

# print( q.x, q.y, q.z, q.w )