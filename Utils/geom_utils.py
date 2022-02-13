import numpy as np
import cv2
from sklearn import linear_model

def umeyama_alignment(x, y, with_scale=True):
    """
    Computes the least squares solution parameters of an Sim(m) matrix
    that minimizes the distance between a set of registered points.
    Umeyama, Shinji: Least-squares estimation of transformation parameters
                     between two point patterns. IEEE PAMI, 1991
    :param x: mxn matrix of points, m = dimension, n = nr. of data points
    :param y: mxn matrix of points, m = dimension, n = nr. of data points
    :param with_scale: set to True to align also the scale (default: 1.0 scale)
    :return: r, t, c - rotation matrix, translation vector and scale factor
    """
    if x.shape != y.shape:
        assert False, "x.shape not equal to y.shape"

    # m = dimension, n = nr. of data points
    m, n = x.shape

    # means, eq. 34 and 35
    mean_x = x.mean(axis=1)
    mean_y = y.mean(axis=1)

    # variance, eq. 36
    # "transpose" for column subtraction
    sigma_x = 1.0 / n * (np.linalg.norm(x - mean_x[:, np.newaxis])**2)

    # covariance matrix, eq. 38
    outer_sum = np.zeros((m, m))
    for i in range(n):
        outer_sum += np.outer((y[:, i] - mean_y), (x[:, i] - mean_x))
    cov_xy = np.multiply(1.0 / n, outer_sum)

    # SVD (text betw. eq. 38 and 39)
    u, d, v = np.linalg.svd(cov_xy)

    # S matrix, eq. 43
    s = np.eye(m)
    if np.linalg.det(u) * np.linalg.det(v) < 0.0:
        # Ensure a RHS coordinate system (Kabsch algorithm).
        s[m - 1, m - 1] = -1

    # rotation, eq. 40
    r = u.dot(s).dot(v)

    # scale & translation, eq. 42 and 41
    c = 1 / sigma_x * np.trace(np.diag(d).dot(s)) if with_scale else 1.0
    t = mean_y - np.multiply(c, r.dot(mean_x))

    return r, t, c


def unprojection_kp(kp, kp_depth, cam_intrinsics):
    """Convert kp to XYZ
    Args:
        kp (Nx2 array): [x, y] keypoints
        kp_depth (Nx2 array): keypoint depth
        cam_intrinsics (Intrinsics): camera intrinsics
    Returns:
        XYZ (Nx3): 3D coordinates
    """
    N = kp.shape[0]
    # initialize regular grid
    XYZ = np.ones((N, 3, 1))
    XYZ[:, :2, 0] = kp
    
    inv_K = np.ones((1, 3, 3))
    inv_K[0] = np.linalg.inv(cam_intrinsics)
    inv_K = np.repeat(inv_K, N, axis=0)

    XYZ = np.matmul(inv_K, XYZ)[:, :, 0]
    XYZ[:, 0] = XYZ[:, 0] * kp_depth
    XYZ[:, 1] = XYZ[:, 1] * kp_depth
    XYZ[:, 2] = XYZ[:, 2] * kp_depth
    return XYZ

def normalize_kp(kp, cam_intr):
    kp_norm = kp.copy()
    kp_norm[:, 0] = \
        (kp[:, 0] - cam_intr[0,2]) / cam_intr[0,0]
    kp_norm[:, 1] = \
        (kp[:, 1] - cam_intr[1,2]) / cam_intr[1,1]

    kp1_3D = np.ones((3, kp_norm.shape[0]))
    kp1_3D[0], kp1_3D[1] = kp_norm[:, 0].copy(), kp_norm[:, 1].copy()

    return kp1_3D

def triangulation(kp1, kp2, T_1w, T_2w, cam_intr):
    """Triangulation to get 3D points
    Args:
        kp1 (Nx2): keypoint in view 1 (normalized)
        kp2 (Nx2): keypoints in view 2 (normalized)
        T_1w (4x4): pose of view 1 w.r.t  i.e. T_1w (from w to 1)
        T_2w (4x4): pose of view 2 w.r.t world, i.e. T_2w (from w to 2)
    Returns:
        X (3xN): 3D coordinates of the keypoints w.r.t world coordinate
        X1 (3xN): 3D coordinates of the keypoints w.r.t view1 coordinate
        X2 (3xN): 3D coordinates of the keypoints w.r.t view2 coordinate
    """
    
    kp1_norm = kp1.copy()
    kp2_norm = kp2.copy()

    kp1_norm[:, 0] = \
        (kp1[:, 0] - cam_intr[0,2]) / cam_intr[0,0]
    kp1_norm[:, 1] = \
        (kp1[:, 1] - cam_intr[1,2]) / cam_intr[1,1]
    kp2_norm[:, 0] = \
        (kp2[:, 0] - cam_intr[0,2]) / cam_intr[0,0]
    kp2_norm[:, 1] = \
        (kp2[:, 1] - cam_intr[1,2]) / cam_intr[1,1]

    kp1_3D = np.ones((3, kp1_norm.shape[0]))
    kp2_3D = np.ones((3, kp2_norm.shape[0]))
    kp1_3D[0], kp1_3D[1] = kp1_norm[:, 0].copy(), kp1_norm[:, 1].copy()
    kp2_3D[0], kp2_3D[1] = kp2_norm[:, 0].copy(), kp2_norm[:, 1].copy()

    X = cv2.triangulatePoints(T_1w[:3], T_2w[:3], kp1_3D[:2], kp2_3D[:2])
    X /= X[3]
    X1 = T_1w[:3] @ X
    X2 = T_2w[:3] @ X
    return X[:3].T, X1, X2


def transform_points(points_3D, pose):
  points_4D = np.c_[ points_3D,  np.ones(len(points_3D)) ].T
  transformed_points_4D = pose @ points_4D
  points_3D = transformed_points_4D[:3].T
  return points_3D



def find_scale(f13D, f23D):
    '''
    Finds relative scale, given 2 sets of 3D points, by getting the ratio of relative distances between corresponding points.
    Args:
        f13D : estimated 3D points, whose scale needs to be found.
        f23D : Reference 3D points.
    Returns:
        Scale.
    '''

    valid_1 = f13D[...,2] > 0
    valid_2 = f23D[...,2] > 0
    valid = valid_1 & valid_2
    f13D = f13D[valid]
    f23D = f23D[valid]
    indices = np.random.choice(np.arange(0,len(f23D)), size=(5 * len(f23D),2),replace=True)
    # no_of_pts = len(f13D)
    # grid = np.mgrid[:no_of_pts,:no_of_pts].reshape(2, -1).T
    # indices = grid.take(np.random.choice(grid.shape[0], no_of_pts * 5, replace=False), axis=0)

    # for indic in indices[::3]:
    #   print(indic)
    # done
    indices = indices[indices[...,0]!=indices[...,1]]
    num = np.linalg.norm(f13D[indices[...,0]] - f13D[indices[...,1]], axis=1).reshape((len(indices),1))
    den = np.linalg.norm(f23D[indices[...,0]] - f23D[indices[...,1]], axis=1).reshape((len(indices),1))
    ransac = linear_model.RANSACRegressor(
                base_estimator=linear_model.LinearRegression(
                    fit_intercept=False),
                min_samples=2,
                max_trials=100,
                stop_probability=0.99,
                residual_threshold=1.0
                                        )
    ransac.fit(
            num,
            den
            )
    scale = ransac.estimator_.coef_[0, 0]
    return scale
    return np.median(den/num)

