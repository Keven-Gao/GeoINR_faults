from sklearn.decomposition import PCA
import numpy as np

def ellipsoid_parameters(points, margin=[1,1,1], scale_normal=1):
    """
    Calculate the parameters of the ellipsoid
    """
    # 1. Compute the centroid of the point cloud
    centroid = np.mean(points, axis=0)
    points_centered = points - centroid

    # 2. Perform PCA on the centered points to get principal directions
    pca = PCA(n_components=3)
    pca.fit(points_centered)
    principal_axes = pca.components_  # shape (3, 3), each row is a unit vector along a principal axis
    # The third principal component (index 2) is the smallest variance direction, treat it as the normal vector
    normal_vector = principal_axes[2]

    # 3. Determine axis lengths (semi-axes) for ellipsoid along each principal direction
    # Project points onto each principal axis to find the extent in that direction
    # We can use PCA transform for convenience, which gives coordinates in the principal axis basis
    points_transformed = pca.transform(points_centered)  # shape (N, 3), coordinates in PCA space
    # Compute the half-length needed along each axis: max absolute coordinate value along that axis
    max_extent = np.max(np.abs(points_transformed), axis=0)  # length 3 array
    # Apply margin to each extent
    axes_lengths = max_extent * margin

    axes_lengths[2] = axes_lengths[2] * scale_normal  

    if normal_vector[2] < 0:
        normal_vector = -normal_vector

    return centroid, normal_vector, axes_lengths


def normal_to_azimuth_dip_wrong(normal_vector):
    """
    Transfer normal_vector n = (nx, ny, nz) to azimuth (alpha) and dip (delta).
      - alpha (dip direction, azimuth: 0~360°)
      - delta (dip angle, dip: 0~90°)
    Coordinate system: +y point to North, +x point to East, +z point to Up.
    """
    nx, ny, nz = normal_vector

    # 1. 计算倾角 δ（单位：度）
    # 注意 arctan2 返回值在 (-180°,180°]
    delta = np.degrees(np.arctan2(np.sqrt(nx**2 + ny**2), nz))
    # 规范化到 [0,90]
    if delta < 0:
        delta += 180
    if delta > 90:
        delta = 180 - delta

    # 2. 计算下坡方向在水平面的投影 h
    if nz < 0:
        hx, hy = -nx, -ny
    else:
        hx, hy =  nx,  ny

    # 3. 计算倾向 α（单位：度），以正北为0，顺时针增大
    alpha = np.degrees(np.arctan2(hx, hy)) % 360

    return alpha, delta


def normal_to_azimuth_dip(normal_vector):
    '''
    Transform the normal_vector n = (nx, ny, nz) to dip direction angle (phi) and dip angle (theta).
    '''
    u_j = np.array([0, 1, 0])  # Up vector in global coordinates
    u_k = np.array([0, 0, 1])  # North vector in global coordinates
    v = np.array(normal_vector) # Normal vector in global coordinates
    if v[2] < 0:
        v = -v
    u_v = np.array([v[0], v[1], 0])  # Project normal onto horizontal plane

    # calculate dip direction angle (phi)
    if v[0] >= 0: 
        dot_phi = np.dot(u_v, u_j)
        norm_u_v = np.linalg.norm(u_v)
        norm_u_j = np.linalg.norm(u_j)
        cos_phi = dot_phi / (norm_u_v * norm_u_j)
        phi_rad = np.arccos(cos_phi)
        phi_deg = phi_rad * 180 / np.pi
    elif v[0] < 0: 
        dot_phi = np.dot(u_v, u_j)
        norm_u_v = np.linalg.norm(u_v)
        norm_u_j = np.linalg.norm(u_j)
        cos_phi = dot_phi / (norm_u_v * norm_u_j)
        phi_rad = np.arccos(cos_phi)
        phi_deg = 360 - (phi_rad * 180 / np.pi)  
    else:
        raise ValueError("Invalid normal vector: {}".format(normal_vector))

    # calculate dip angle (theta)
    dot_theta = np.dot(v, u_k)
    norm_v = np.linalg.norm(v)
    norm_u_k = np.linalg.norm(u_k)
    cos_theta = dot_theta / (norm_v * norm_u_k)
    theta_rad = np.arccos(cos_theta)
    theta_deg = theta_rad * 180 / np.pi

    return phi_deg, theta_deg

