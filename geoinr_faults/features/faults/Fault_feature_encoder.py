import numpy as np
from scipy.interpolate import RBFInterpolator
import pyvista as pv


class FiniteFault:
    """
    Class to apply fault feature deformations on 3D point clouds.

    Methods:
        rotation_matrix: compute rotation matrix from global to fault-local coords
        translate_to_local: translate global points to fault-centered local coords
        rotate_points: apply rotation matrix to points
        displacement_field_d: compute planar displacement magnitude field
        displacement_Dy: compute Y-axis displacement in local coords
        displacement_Dz: compute Z-axis displacement in local coords
        compute_rbf_interpolator: train RBF on fault trace in local coords
        apply_deformation: full pipeline to deform input points
    """

    @staticmethod
    def rotation_matrix(theta: float, phi: float) -> np.ndarray:
        """
        Compute the rotation matrix R (3x3) to transform global coordinates
        into fault-local coordinates using strike (theta) and dip (phi).

        Args:
            theta: strike angle in radians (rotation about Z-axis)
            phi: dip angle in radians (rotation about new X-axis)

        Returns:
            R: 3x3 orthonormal rotation matrix
        """
        # Combined rotation: first about Z (strike), then about new X (dip)
        R = np.array([
            [ np.cos(theta),           -np.sin(theta),            0       ],
            [ np.sin(theta)*np.cos(phi),  np.cos(theta)*np.cos(phi), -np.sin(phi) ],
            [ np.sin(theta)*np.sin(phi),  np.cos(theta)*np.sin(phi),  np.cos(phi) ]
        ])
        return R

    @staticmethod
    def translate_to_local(points: np.ndarray, center: np.ndarray) -> np.ndarray:
        """
        Translate global points so that the fault center becomes the origin.

        Args:
            points: (N,3) array of global coordinates
            center: length-3 array [X0, Y0, Z0] for fault center

        Returns:
            translated: (N,3) array in translated local frame
        """
        return points - center[np.newaxis, :]

    @staticmethod
    def rotate_points(points: np.ndarray, R: np.ndarray) -> np.ndarray:
        """
        Apply rotation matrix to points.

        Args:
            points: (N,3) array in local-translated frame
            R: (3,3) rotation matrix

        Returns:
            rotated: (N,3) array in fault-local coordinates
        """
        return (R.dot(points.T)).T

    @staticmethod
    def displacement_field_d(
        x_loc: np.ndarray,
        y_loc: np.ndarray,
        lx: float,
        ly: float,
        dmax: float,
        x0: float,
        y0: float
    ) -> np.ndarray:
        """
        Compute planar (x-y) displacement magnitude field d_xy.
        Uses an exponential tapering around a center (x0,y0).
        Negative values are truncated to zero.
        """
        r_xy = np.sqrt(((x_loc - x0)/lx)**2 + ((y_loc - y0)/ly)**2)
        d_xy = 2 * dmax * (1 - r_xy) * np.exp(((1 + r_xy)/2)**2 - r_xy**2)
        return np.where(d_xy >= 0, d_xy, 0.0)

    @staticmethod
    def displacement_Dy(
        x_loc: np.ndarray,
        y_loc: np.ndarray,
        z_loc: np.ndarray,
        lx: float,
        ly: float,
        lz: float,
        dmax: float,
        x0: float,
        y0: float,
        lam: float,
        f_xy: np.ndarray
    ) -> np.ndarray:
        """
        Compute displacement along local Y-axis (dip-direction) with depth decay.

        Args:
            x_loc, y_loc, z_loc: local coordinates arrays
            lx, ly, lz: characteristic lengths
            dmax: max planar displacement
            x0, y0: planar center offsets
            lam: partitioning factor between hanging and footwall
            f_xy: fault surface z-values at (x_loc,y_loc)

        Returns:
            dy: array of Y displacements
        """
        # raw decay factor: linear with vertical distance to fault surface
        alpha_raw = 1 - np.abs(z_loc - f_xy) / lz
        alpha = np.where(alpha_raw >= 0, alpha_raw**2, 0.0)

        # compute planar displacement field once
        d_xy = FiniteFault.displacement_field_d(
            x_loc, y_loc, lx, ly, dmax, x0, y0
        )

        # initialize output
        dy = np.zeros_like(z_loc)

        # Hanging-wall mask: points above fault surface within decay thickness
        mask_hang = (z_loc >= f_xy) & (z_loc <= f_xy + lz)
        # Footwall mask: points below fault surface within decay thickness
        mask_foot = (z_loc >= f_xy - lz) & (z_loc <= f_xy)

        # apply lam and (lam-1) factors
        dy[mask_hang] = lam * d_xy[mask_hang] * alpha[mask_hang]
        dy[mask_foot] = (lam - 1) * d_xy[mask_foot] * alpha[mask_foot]
        return dy

    @staticmethod
    def displacement_Dz(
        f_xy: np.ndarray,
        f_xy_Dy: np.ndarray
    ) -> np.ndarray:
        """
        Compute Z-axis displacement as difference between displaced and original fault surface.

        Args:
            f_xy: original fault surface z-values
            f_xy_Dy: fault-surface z after Y-displacement

        Returns:
            dz: array of Z displacements
        """
        return f_xy_Dy - f_xy

    @staticmethod
    def compute_rbf_interpolator(
        fault_points_local: np.ndarray,
        kernel: str = 'cubic',
        epsilon: float = 1.0,
        smoothing: float = 0.0
    ) -> RBFInterpolator:
        """
        Train an RBF interpolator on the fault trace in local coordinates.

        Args:
            fault_points_local: (M,3) points on fault plane
            kernel: RBF kernel name
            epsilon: RBF parameter
            smoothing: smoothing factor

        Returns:
            rbf_interp: trained RBFInterpolator mapping (x,y)->z
        """
        xy = fault_points_local[:, :2]
        z = fault_points_local[:, 2]
        return RBFInterpolator(xy, z, kernel=kernel, epsilon=epsilon, smoothing=smoothing)


    @staticmethod
    def hanging_wall_mask(
        points: np.ndarray,
        fault_points: np.ndarray,
        theta: float,
        phi: float,
        X0: float,
        Y0: float,
        Z0: float,
        rbf_kwargs: dict = None
    ) -> np.ndarray:
        """
        Compute a boolean mask of points lying on the hanging wall of a fault.
        Hanging wall defined by local z >= fault surface f_xy(x,y).
        """
        R = FiniteFault.rotation_matrix(theta, phi)
        center = np.array([X0, Y0, Z0])
        local = FiniteFault.rotate_points(
            FiniteFault.translate_to_local(points, center), R
        )
        fault_local = FiniteFault.rotate_points(
            FiniteFault.translate_to_local(fault_points, center), R
        )
        rbf = FiniteFault.compute_rbf_interpolator(fault_local, **(rbf_kwargs or {}))
        f_xy = rbf(local[:, :2])

        return local[:, 2] >= f_xy

    
    @staticmethod
    def apply_truncation(
        points: np.ndarray,
        fault_points: np.ndarray,
        theta: float,
        phi: float,
        X0: float,
        Y0: float,
        Z0: float,
        rbf_kwargs: dict = None
    ) -> np.ndarray:
        R = FiniteFault.rotation_matrix(theta, phi)
        center = np.array([X0, Y0, Z0])
        local = FiniteFault.rotate_points(
            FiniteFault.translate_to_local(points, center), R
        )
        fault_local = FiniteFault.rotate_points(
            FiniteFault.translate_to_local(fault_points, center), R
        )
        rbf = FiniteFault.compute_rbf_interpolator(fault_local, **(rbf_kwargs or {}))
        f_xy = rbf(local[:, :2])
        mask_pen = local[:, 2] < f_xy
        if not np.any(mask_pen):
            return points.copy()
        local[mask_pen, 2] = f_xy[mask_pen]
        corrected = FiniteFault.rotate_points(local, R.T) + center
        out = points.copy()
        out[mask_pen] = corrected[mask_pen]
        return out


    @staticmethod
    def fault_features(
        points: np.ndarray,
        fault_points: np.ndarray,
        theta: float,
        phi: float,
        X0: float,
        Y0: float,
        Z0: float,
        lx: float,
        ly: float,
        lz: float,
        dmax: float,
        lam: float,
        x0: float = 0.0,
        y0: float = 0.0,
        rbf_kwargs: dict = None,
        Normal_fault: bool = True,
        truncation: bool = False,
        trunc_fault: dict = None,
        hanging_trunc: bool = False
    ) -> np.ndarray:
        """
        Using the moving distances (D_y and D_z) as the fault feature,

        Args:
            points: (N,3) array of global coordinates
            fault_points: (M,3) array of global fault trace coords
            theta, phi: rotation angles (radians)
            X0, Y0, Z0: fault center in global coords
            lx, ly, lz: characteristic lengths
            dmax: max planar displacement
            x0, y0: planar center offsets
            lam: partition factor
            rbf_kwargs: kwargs for RBFInterpolator
            hanging_trunc: bool, if True, means the hanging wall of older fault do not affect by the younger fault

        Returns:
            deformed_global: (N,3) deformed points in global frame
        """
        # prepare rotation and translation
        R = FiniteFault.rotation_matrix(theta, phi)
        center = np.array([X0, Y0, Z0])

        # translate and rotate input points to local frame
        pts_local = FiniteFault.rotate_points(
            FiniteFault.translate_to_local(points, center), R
        )
        x_loc, y_loc, z_loc = pts_local.T

        # translate and rotate fault_points for RBF
        fault_local = FiniteFault.rotate_points(
            FiniteFault.translate_to_local(fault_points, center), R
        )

        # train RBF on fault plane
        rbf_kwargs = rbf_kwargs or {}
        rbf = FiniteFault.compute_rbf_interpolator(fault_local, **rbf_kwargs)
        # evaluate original fault surface
        f_xy = rbf(pts_local[:, :2])

        # compute Dy and Dz
        D_y = FiniteFault.displacement_Dy(
            x_loc, y_loc, z_loc,
            lx, ly, lz, dmax, x0, y0, lam, f_xy
        )
        new_y_loc = y_loc + D_y
        f_xy_Dy = rbf(np.column_stack([x_loc, new_y_loc]))
        D_z = FiniteFault.displacement_Dz(f_xy, f_xy_Dy)

        # apply displacements
        pts_def_local = np.vstack([x_loc,
                                   new_y_loc,
                                   z_loc + D_z]).T

        # rotate back and translate to global
        deformed_points = FiniteFault.rotate_points(pts_def_local, R.T)
        deformed_points += center[np.newaxis, :]

        # calculate the deformation magnitude
        deformed_tol = np.sqrt(D_y**2 + D_z**2)
        deformed_tol_signed = np.sign(D_y) * deformed_tol

        if truncation and trunc_fault:
            hang_mask = FiniteFault.hanging_wall_mask(
                points,
                trunc_fault['fault_points'],
                trunc_fault['theta'], trunc_fault['phi'],
                trunc_fault['X0'], trunc_fault['Y0'], trunc_fault['Z0'],
                trunc_fault.get('rbf_kwargs'),
                )
            # set deformed_tol_signed to zero for points not in truncation side
            if hanging_trunc:
                deformed_tol_signed[hang_mask] = 0.0
            else:
                deformed_tol_signed[~hang_mask] = 0.0

        if Normal_fault:
            deformed_tol_signed = deformed_tol_signed * (-1)  # make sure the move down is negative

        return deformed_tol_signed


    @staticmethod
    def deformed_coordinates(
        points: np.ndarray,
        fault_points: np.ndarray,
        theta: float,
        phi: float,
        X0: float,
        Y0: float,
        Z0: float,
        lx: float,
        ly: float,
        lz: float,
        dmax: float,
        lam: float,
        x0: float = 0.0,
        y0: float = 0.0,
        rbf_kwargs: dict = None,
        truncation: bool = False,
        trunc_fault: dict = None,
        hanging_trunc: bool = False
    ) -> np.ndarray:
        """
        Output deformed coordinates of points, assigning this coordinates back to the PyVista meshgrid and replace the old coordinates, 
        which can use for display the movie of geological deformation.

        Args:
            points: (N,3) array of global coordinates
            fault_points: (M,3) array of global fault trace coords
            theta, phi: rotation angles (radians)
            X0, Y0, Z0: fault center in global coords
            lx, ly, lz: characteristic lengths
            dmax: max planar displacement
            x0, y0: planar center offsets
            lam: partition factor
            rbf_kwargs: kwargs for RBFInterpolator
            hanging_trunc: bool, if True, means the hanging wall of older fault do not affect by the younger fault

        Returns:
            deformed_global: (N,3) deformed points in global frame
        """
        # prepare rotation and translation
        R = FiniteFault.rotation_matrix(theta, phi)
        center = np.array([X0, Y0, Z0])

        # translate and rotate input points to local frame
        pts_local = FiniteFault.rotate_points(
            FiniteFault.translate_to_local(points, center), R
        )
        x_loc, y_loc, z_loc = pts_local.T

        # translate and rotate fault_points for RBF
        fault_local = FiniteFault.rotate_points(
            FiniteFault.translate_to_local(fault_points, center), R
        )

        # train RBF on fault plane
        rbf_kwargs = rbf_kwargs or {}
        rbf = FiniteFault.compute_rbf_interpolator(fault_local, **rbf_kwargs)
        # evaluate original fault surface
        f_xy = rbf(pts_local[:, :2])

        # compute Dy and Dz
        D_y = FiniteFault.displacement_Dy(
            x_loc, y_loc, z_loc,
            lx, ly, lz, dmax, x0, y0, lam, f_xy
        )
        new_y_loc = y_loc + D_y
        f_xy_Dy = rbf(np.column_stack([x_loc, new_y_loc]))
        D_z = FiniteFault.displacement_Dz(f_xy, f_xy_Dy)

        # apply displacements
        pts_def_local = np.vstack([x_loc,
                                   new_y_loc,
                                   z_loc + D_z]).T

        # rotate back and translate to global
        deformed_points = FiniteFault.rotate_points(pts_def_local, R.T)
        deformed_points += center[np.newaxis, :]

        if truncation and trunc_fault:
            hang_mask = FiniteFault.hanging_wall_mask(
                points,
                trunc_fault['fault_points'],
                trunc_fault['theta'], trunc_fault['phi'],
                trunc_fault['X0'], trunc_fault['Y0'], trunc_fault['Z0'],
                trunc_fault.get('rbf_kwargs')
                )
            
            # the un-truncation side maintains the original coordinates
            if hanging_trunc:
                deformed_points[hang_mask] = points[hang_mask]
            else:
                deformed_points = FiniteFault.apply_truncation(
                    deformed_points,
                    trunc_fault['fault_points'],
                    trunc_fault['theta'], trunc_fault['phi'],
                    trunc_fault['X0'], trunc_fault['Y0'], trunc_fault['Z0'],
                    trunc_fault.get('rbf_kwargs')
                )
                deformed_points[~hang_mask] = points[~hang_mask]

        return deformed_points
    

    @staticmethod
    def compute_fault_surface(
        theta: float, phi: float,
        X0: float, Y0: float, Z0: float,
        lx: float, ly: float, lz: float,
        rbf: RBFInterpolator,
        x_range: tuple = (-1, 1),
        y_range: tuple = (-1, 1),
        resolution: int = 50,
        points: np.ndarray = None
    ) -> pv.StructuredGrid:
        """
        Compute and return the clipped ellipsoidal fault surface mesh in global coords.

        Args:
            theta, phi: rotation angles
            X0, Y0, Z0: fault center global coords
            lx, ly, lz: characteristic lengths for ellipsoid
            rbf: trained RBFInterpolator mapping (x,y)->z in local coords
            x_range, y_range: ranges for grid (local coords)
            resolution: number of steps per axis

        Returns:
            pv.StructuredGrid of clipped fault surface in global frame
        """
        # generate grid in local f-plane coords
        xs = np.linspace(x_range[0], x_range[1], resolution)
        ys = np.linspace(y_range[0], y_range[1], resolution)
        xx, yy = np.meshgrid(xs, ys)
        pts2d = np.column_stack([xx.ravel(), yy.ravel()])
        zz = rbf(pts2d).reshape(xx.shape)
        pts3d = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        # build structured grid
        surf = pv.StructuredGrid()
        surf.points = pts3d
        surf.dimensions = [resolution, resolution, 1]
        # ellipsoid function for clipping
        ell = (pts3d[:,0]**2/lx**2) + (pts3d[:,1]**2/ly**2) + (pts3d[:,2]**2/lz**2) - 1.0
        surf.point_data['ellipsoid_func'] = ell
        clipped = surf.clip_scalar('ellipsoid_func', value=0.0, invert=True)
        # transform back to global
        R = FiniteFault.rotation_matrix(theta, phi)
        center = np.array([X0, Y0, Z0])
        pts_clip = clipped.points
        pts_glob = (R.T.dot(pts_clip.T)).T + center[np.newaxis, :]
        clipped.points = pts_glob
        # compute model boundary box and clip
        bounds = [
            points[:,0].min(), points[:,0].max(),
            points[:,1].min(), points[:,1].max(),
            points[:,2].min(), points[:,2].max()
        ]
        model_boundary = pv.Box(bounds=bounds)
        clipped = clipped.clip_box(model_boundary, invert=False)
        return clipped
    
    @staticmethod
    def fault_mesh(
        points: np.ndarray,
        fault_points: np.ndarray,
        theta: float,
        phi: float,
        X0: float,
        Y0: float,
        Z0: float,
        lx: float,
        ly: float,
        lz: float,
        rbf_kwargs: dict = None,
        truncation: bool = False,
        trunc_fault_mesh: pv.PolyData = None,
        hanging_trunc: bool = False,
        resolution: int = 100
    ) -> np.ndarray:
        """
        Generate fault mesh.

        Args:
            points: (N,3) array of global coordinates
            fault_points: (M,3) array of global fault trace coords
            theta, phi: rotation angles (radians)
            X0, Y0, Z0: fault center in global coords
            lx, ly, lz: characteristic lengths
            dmax: max planar displacement
            x0, y0: planar center offsets
            lam: partition factor
            rbf_kwargs: kwargs for RBFInterpolator
            resolution: number of steps per axis for mesh grid

        Returns:
            deformed_global: (N,3) deformed points in global frame
        """
        # prepare rotation and translation
        R = FiniteFault.rotation_matrix(theta, phi)
        center = np.array([X0, Y0, Z0])

        # translate and rotate fault_points for RBF
        fault_local = FiniteFault.rotate_points(
            FiniteFault.translate_to_local(fault_points, center), R
        )

        # train RBF on fault plane
        rbf_kwargs = rbf_kwargs or {}
        rbf = FiniteFault.compute_rbf_interpolator(fault_local, **rbf_kwargs)

        # compute fault surface
        fault_mesh = FiniteFault.compute_fault_surface(
            theta, phi, X0, Y0, Z0,
            lx, ly, lz, rbf,
            x_range=(-lx*1.1, lx*1.1), y_range=(-ly*1.1, ly*1.1), resolution=resolution, points= points  # *1.1 as safety margin for clipping
        )

        if truncation and trunc_fault_mesh:
            if hanging_trunc:
                # if hanging truncation, only keep points above truncation surface
                fault_mesh = fault_mesh.clip_surface(
                    trunc_fault_mesh,
                    invert=True)
            else:
                # if footwall truncation, keep points below truncation surface
                fault_mesh = fault_mesh.clip_surface(
                    trunc_fault_mesh,
                    invert=False)

        return fault_mesh
    


class InfiniteFault:
    """
    Class to apply fault feature deformations on 3D point clouds.

    Methods:
        rotation_matrix: compute rotation matrix from global to fault-local coords
        translate_to_local: translate global points to fault-centered local coords
        rotate_points: apply rotation matrix to points
        displacement_field_d: compute planar displacement magnitude field
        displacement_Dy: compute Y-axis displacement in local coords
        displacement_Dz: compute Z-axis displacement in local coords
        compute_rbf_interpolator: train RBF on fault trace in local coords
        apply_deformation: full pipeline to deform input points
    """

    @staticmethod
    def rotation_matrix(theta: float, phi: float) -> np.ndarray:
        """
        Compute the rotation matrix R (3x3) to transform global coordinates
        into fault-local coordinates using strike (theta) and dip (phi).

        Args:
            theta: strike angle in radians (rotation about Z-axis)
            phi: dip angle in radians (rotation about new X-axis)

        Returns:
            R: 3x3 orthonormal rotation matrix
        """
        # Combined rotation: first about Z (strike), then about new X (dip)
        R = np.array([
            [ np.cos(theta),           -np.sin(theta),            0       ],
            [ np.sin(theta)*np.cos(phi),  np.cos(theta)*np.cos(phi), -np.sin(phi) ],
            [ np.sin(theta)*np.sin(phi),  np.cos(theta)*np.sin(phi),  np.cos(phi) ]
        ])
        return R
    
    @staticmethod
    def translate_to_local(points: np.ndarray, center: np.ndarray) -> np.ndarray:
        """
        Translate global points so that the fault center becomes the origin.

        Args:
            points: (N,3) array of global coordinates
            center: length-3 array [X0, Y0, Z0] for fault center

        Returns:
            translated: (N,3) array in translated local frame
        """
        return points - center[np.newaxis, :]
    
    @staticmethod
    def rotate_points(points: np.ndarray, R: np.ndarray) -> np.ndarray:
        """
        Apply rotation matrix to points.

        Args:
            points: (N,3) array in local-translated frame
            R: (3,3) rotation matrix

        Returns:
            rotated: (N,3) array in fault-local coordinates
        """
        return (R.dot(points.T)).T

    @staticmethod
    def compute_rbf_interpolator_global(
        fault_points_local: np.ndarray,
        kernel: str = 'cubic',
        epsilon: float = 1.0,
        smoothing: float = 0.0
    ) -> RBFInterpolator:
        """
        Train an RBF interpolator on the fault trace in local coordinates.

        Args:
            fault_points_local: (M,3) points on fault plane
            kernel: RBF kernel name
            epsilon: RBF parameter
            smoothing: smoothing factor

        Returns:
            rbf_interp: trained RBFInterpolator mapping (x,y)->z
        """
        yz = fault_points_local[:, 1:]
        x = fault_points_local[:, 0]
        return RBFInterpolator(yz, x, kernel=kernel, epsilon=epsilon, smoothing=smoothing)
    
    def compute_rbf_interpolator_local(
        fault_points_local: np.ndarray,
        kernel: str = 'cubic',
        epsilon: float = 1.0,
        smoothing: float = 0.0
    ) -> RBFInterpolator:
        """
        Train an RBF interpolator on the fault trace in local coordinates.

        Args:
            fault_points_local: (M,3) points on fault plane
            kernel: RBF kernel name
            epsilon: RBF parameter
            smoothing: smoothing factor

        Returns:
            rbf_interp: trained RBFInterpolator mapping (x,y)->z
        """
        xy = fault_points_local[:, :2]
        z = fault_points_local[:, 2]
        return RBFInterpolator(xy, z, kernel=kernel, epsilon=epsilon, smoothing=smoothing)
    
    @staticmethod
    def fault_features(
        points: np.ndarray,
        fault_points: np.ndarray,
        rbf_kwargs: dict = None,
        Normal_fault: bool = True
    ) -> np.ndarray:
        """
        Using the moving distances (D_y and D_z) as the fault feature,

        Args:
            points: (N,3) array of global coordinates
            fault_points: (M,3) array of global fault trace coords
            theta, phi: rotation angles (radians)
            X0, Y0, Z0: fault center in global coords
            lx, ly, lz: characteristic lengths
            dmax: max planar displacement
            x0, y0: planar center offsets
            lam: partition factor
            rbf_kwargs: kwargs for RBFInterpolator

        Returns:
            deformed_global: (N,3) deformed points in global frame
        """
        # train RBF on fault plane
        rbf_kwargs = rbf_kwargs or {}
        rbf = InfiniteFault.compute_rbf_interpolator_global(fault_points, **rbf_kwargs)
        # evaluate original fault surface
        f_xy = rbf(points[:, 1:])
        # Hanging-wall mask: points above fault surface within decay thickness
        mask_hang = (points[:,0] >= f_xy) 
        # Footwall mask: points below fault surface within decay thickness
        mask_foot = (points[:,0] < f_xy)

        fault_features = np.ones(points.shape[0])
        # apply lam and (lam-1) factors
        fault_features[mask_hang] = fault_features[mask_hang]
        fault_features[mask_foot] = fault_features[mask_foot]*(-1)

        if Normal_fault:
            fault_features = fault_features * (-1)
        
        return fault_features

    @staticmethod
    def deformed_coordinates(
        points: np.ndarray,
        fault_points: np.ndarray,
        theta: float,
        phi: float,
        dist: float,
        rbf_kwargs: dict = None,
        Normal_fault: bool = True
    ) -> np.ndarray:
        """
        Output deformed coordinates of points, assigning this coordinates back to the PyVista meshgrid and replace the old coordinates, 
        which can use for display the movie of geological deformation.

        Args:
            points: (N,3) array of global coordinates
            fault_points: (M,3) array of global fault trace coords
            theta, phi: rotation angles (radians)
            dist: displacement
            lam: partition factor
            rbf_kwargs: kwargs for RBFInterpolator

        Returns:
            deformed_global: (N,3) deformed points in global frame
        """
        # prepare rotation and translation
        R = InfiniteFault.rotation_matrix(theta, phi)

        # translate and rotate input points to local frame
        pts_local = InfiniteFault.rotate_points(
            points, R
        )
        x_loc, y_loc, z_loc = pts_local.T

        # translate and rotate fault_points for RBF
        fault_local = InfiniteFault.rotate_points(
            fault_points, R
        )

        # train RBF on fault plane
        rbf_kwargs = rbf_kwargs or {}
        rbf = InfiniteFault.compute_rbf_interpolator_local(fault_local, **rbf_kwargs)
        # evaluate original fault surface
        f_xy = rbf(pts_local[:, :2])

        # compute Dy
        mask_hang = (z_loc >= f_xy)
        # initialize output
        D_y = np.zeros_like(z_loc)
        if Normal_fault: 
            D_y[mask_hang] = dist * np.ones_like(y_loc)[mask_hang]
        else:
            D_y[mask_hang] = -dist * np.ones_like(y_loc)[mask_hang]
        
        # compute Dz
        new_y_loc = y_loc + D_y
        f_xy_Dy = rbf(np.column_stack([x_loc, new_y_loc]))
        D_z = f_xy_Dy - f_xy

        # apply displacements
        pts_def_local = np.vstack([x_loc,
                                   new_y_loc,
                                   z_loc + D_z]).T

        # rotate back and translate to global
        deformed_points = InfiniteFault.rotate_points(pts_def_local, R.T)

        return deformed_points

    @staticmethod
    def fault_mesh(
        points: np.ndarray,
        fault_points: np.ndarray,
        resolution: int = 50,
        rbf_kwargs: dict = None
    ) -> np.ndarray:
        """
        Generate fault mesh.

        Args:
            points: (N,3) array of global coordinates
            fault_points: (M,3) array of global fault trace coords
            theta, phi: rotation angles (radians)
            X0, Y0, Z0: fault center in global coords
            lx, ly, lz: characteristic lengths
            dmax: max planar displacement
            x0, y0: planar center offsets
            lam: partition factor
            rbf_kwargs: kwargs for RBFInterpolator

        Returns:
            deformed_global: (N,3) deformed points in global frame
        """
        # train RBF on fault plane
        rbf_kwargs = rbf_kwargs or {}
        rbf = InfiniteFault.compute_rbf_interpolator_global(fault_points, **rbf_kwargs)

        # compute fault surface
        # generate grid in local f-plane coords
        ys = np.linspace(points[:,1].min(), points[:,1].max(), resolution)
        zs = np.linspace(points[:,2].min(), points[:,2].max(), resolution)
        yy, zz = np.meshgrid(ys, zs)
        pts2d = np.column_stack([yy.ravel(), zz.ravel()])
        xx = rbf(pts2d).reshape(yy.shape)
        pts3d = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        # build structured grid
        surf = pv.StructuredGrid()
        surf.points = pts3d
        surf.dimensions = [1, resolution, resolution]
        # compute model boundary box and clip
        bounds = [
            points[:,0].min(), points[:,0].max(),
            points[:,1].min(), points[:,1].max(),
            points[:,2].min(), points[:,2].max()
        ]
        model_boundary = pv.Box(bounds=bounds)
        clipped = surf.clip_box(model_boundary, invert=False)
        return clipped
    

    