import numpy as np

# Camera intrinsic matrix (fx, fy, cx, cy in pixels)
K = np.array([[800,   0, 320],
              [  0, 800, 240],
              [  0,   0,   1]], dtype=float)
# A set of 3-D points in the camera coordinate frame (X, Y, Z) in metres
points_3d = np.array([
    [0.30,  0.15, 3.0],
    [-0.20, 0.10, 2.0],
    [0.00,  0.00, 1.0],
])

def project(points_3d, K):
    # Convert to homogeneous image coords: s*[x,y,1]^T = K * [X,Y,Z]^T
    proj = (K @ points_3d.T).T
    # Perspective divide by Z (the third component before normalisation)
    pixels = proj[:, :2] / points_3d[:, 2:3]
    return pixels

pixel_coords = project(points_3d, K)
print('Projected pixel coordinates:\n', pixel_coords)
