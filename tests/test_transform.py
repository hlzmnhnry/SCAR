import unittest
import numpy as np

from scar import orientation_ned_to_enu, lla_ned_euler_to_enu_pose, lla_ned_quat_to_enu_pose
from scipy.spatial.transform import Rotation
from pyproj import Transformer

class TestTransform(unittest.TestCase):

    def test_orientation_ned_to_enu(self):
        
        angle_x_ned_deg = 0.
        angle_y_ned_deg = 0.
        angle_z_ned_deg = 0.
        
        angle_x_enu_rad, angle_y_enu_rad, angle_z_enu_rad = orientation_ned_to_enu(
            angle_x=np.deg2rad(angle_x_ned_deg),
            angle_y=np.deg2rad(angle_y_ned_deg),
            angle_z=np.deg2rad(angle_z_ned_deg)
        )

        self.assertAlmostEqual(np.rad2deg(angle_x_enu_rad), 180, places=8)
        self.assertAlmostEqual(np.rad2deg(angle_y_enu_rad), 0, places=8)
        self.assertAlmostEqual(np.rad2deg(angle_z_enu_rad), 90, places=8)
        
    def test_lla_ned_euler_to_enu_pose(self):

        latitude = 50.73260057409592
        longitude = 7.092927527775457
        altitude = 724.2397479845032
        angle_x_rad = -0.1198255494236946
        angle_y_rad = -0.0708576366305351
        angle_z_rad = -2.3566126823425293
        
        P, Q = lla_ned_euler_to_enu_pose(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            angle_x_rad=angle_x_rad,
            angle_y_rad=angle_y_rad,
            angle_z_rad=angle_z_rad
        )
        
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
        E, N, U = transformer.transform(longitude, latitude, altitude)

        np.testing.assert_almost_equal(P, np.array([E, N, U]), decimal=3)
        
        _, _, ze = Rotation.from_quat(Q, scalar_first=False).as_euler("xyz", degrees=True)
        
        #                          NED:  0° (N)
        #                          ENU: 90°
        #                              ↑
        #                              |
        #                              |
        #                              |
        #                              |
        # NED: -90° (W) ←--------------●-------------→ NED: 90° (E)
        # ENU: 180°                 CENTER             ENU: 0°
        #                              |
        #                              |
        #                              |
        #                              ↓
        #                          NED: 180° (S)
        #                          ENU: -90°

        # shift ENU yaw back to NED (sign/direction change and 90 deg offset)
        ze_back = 90 - ze
        # wrap back to -180 -> 180 
        ze_back = (ze_back + 180) % 360 - 180

        self.assertAlmostEqual(ze_back, np.rad2deg(angle_z_rad))
        
        latitude = 50.63384047169277
        longitude = 7.210079007461919
        altitude = 682.7629100495932
        angle_x_rad = 0.0150337889790534
        angle_y_rad = 0.0301067847758531
        angle_z_rad = -0.2685998380184173
        
        P, Q = lla_ned_euler_to_enu_pose(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            angle_x_rad=angle_x_rad,
            angle_y_rad=angle_y_rad,
            angle_z_rad=angle_z_rad
        )
        
        transformer = Transformer.from_crs("EPSG:4326", "EPSG:32632", always_xy=True)
        E, N, U = transformer.transform(longitude, latitude, altitude)

        np.testing.assert_almost_equal(P, np.array([E, N, U]), decimal=3)
        
        _, _, ze = Rotation.from_quat(Q, scalar_first=False).as_euler("xyz", degrees=True)
        
        # shift ENU yaw back to NED (sign/direction change and 90 deg offset)
        ze_back = 90 - ze
        # wrap back to -180 -> 180 
        ze_back = (ze_back + 180) % 360 - 180

        self.assertAlmostEqual(ze_back, np.rad2deg(angle_z_rad))

    def test_lla_ned_quat_to_enu_pose(self):
        
        latitude = 50.73260057409592
        longitude = 7.092927527775457
        altitude = 724.2397479845032
        angle_x_rad = -0.1198255494236946
        angle_y_rad = -0.0708576366305351
        angle_z_rad = -2.3566126823425293
        quat_x = -0.0555571988224983
        quat_y = 0.0417651310563087
        quat_z = -0.9225342869758606
        quat_w = 0.3796046674251556
        
        P1, Q1 = lla_ned_euler_to_enu_pose(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            angle_x_rad=angle_x_rad,
            angle_y_rad=angle_y_rad,
            angle_z_rad=angle_z_rad
        )
        
        P2, Q2 = lla_ned_quat_to_enu_pose(
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            quat_x=quat_x,
            quat_y=quat_y,
            quat_z=quat_z,
            quat_w=quat_w
        )
        
        np.testing.assert_almost_equal(P1, P2)
        np.testing.assert_almost_equal(Q1, Q2)

if __name__ == "__main__":
    unittest.main()
