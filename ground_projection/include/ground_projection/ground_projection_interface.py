import os.path

from duckietown_msgs.msg import Segment, SegmentList  # @UnresolvedImport

import cv2

import duckietown_utils as dtu
import numpy as np
from pi_camera import get_camera_info_for_robot

from .configuration import get_homography_for_robot
from .ground_projection_geometry import GroundProjectionGeometry


__all__ = [
    'GroundProjection',
]


class GroundProjection(object):

    def __init__(self, robot_name):
        camera_info = get_camera_info_for_robot(robot_name)
        homography = get_homography_for_robot(robot_name)
        self.gpc = GroundProjectionGeometry(camera_info, homography)
        
        self.board_ = load_board_info()

    def get_camera_info(self):
        return self.gpc.ci
    
    # wait until we have recieved the camera info message through ROS and then initialize
#     def initialize_pinhole_camera_model(self, camera_info):
#         self.ci = camera_info
#         self.pcm.fromCameraInfo(camera_info)
        #print("pinhole camera model initialized")

    def rectify_point(self, p):
        return self.gpc.rectify_point(p)
    
    
    def vector2pixel(self, vec):
        return self.gpc.vector2pixel(vec)

    def pixel2vector(self, pixel):
        return self.gpc.pixel2vector(pixel)

    def vector2ground(self, vec):
        return self.gpc.vector2ground(vec)

    def ground2vector(self, point):
        return self.gpc.ground2vector(point)

    def pixel2ground(self,pixel):
        return self.gpc.pixel2ground(pixel)

    def ground2pixel(self, point):
        return self.gpc.ground2pixel(point)

    def rectify(self, cv_image_raw):
        ''' Undistort image '''
        return self.gpc.rectify(cv_image_raw)

    def estimate_homography(self, cv_image):
        '''Estimate ground projection using instrinsic camera calibration parameters'''
        cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cv_image_rectified = self.rectify(cv_image)
        dtu.logger.info("image rectified")

        ret, corners = cv2.findChessboardCorners(cv_image_rectified, (self.board_['width'], self.board_['height']))
        if ret == False:
            dtu.logger.error("No corners found in image")
            exit(1)
        if len(corners) != self.board_['width'] * self.board_['height']:
            dtu.logger.error("Not all corners found in image")
            exit(2)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.01)
        corners2 = cv2.cornerSubPix(cv_image_rectified, corners, (11,11), (-1,-1), criteria)

        #TODO flip checks
        src_pts = []
        for r in range(self.board_['height']):
            for c in range(self.board_['width']):
                src_pts.append(np.array([r * self.board_['square_size'] , c * self.board_['square_size']] , dtype='float32') + self.board_['offset'])
        # OpenCV labels corners left-to-right, top-to-bottom
        # so we reverse the groud points
        src_pts.reverse()

        # Compute homography from image to ground
        self.H, mask = cv2.findHomography(corners2.reshape(len(corners2), 2), np.array(src_pts), cv2.RANSAC)
        extrinsics_filename = dtu.get_duckiefleet_root() + "/calibrations/camera_extrinsic/" + self.robot_name + ".yaml"
        self.write_homography(extrinsics_filename)
        dtu.logger.info("Wrote ground projection to {}".format(extrinsics_filename))
 
    def write_homography(self, filename):
        ob = {'homography': sum(self.H.reshape(9,1).tolist(),[])}
        dtu.yaml_write_to_file(ob,filename)

    @dtu.contract(sl=SegmentList, returns=SegmentList)
    def find_ground_coordinates(self, sl):
        """ Creates a new segment list with the ground coordinates set """
        
        sl2 = SegmentList()
        sl2.header = sl.header
        
        # Get ground truth of segmentList
        for s1 in sl.segments:
            s2 = Segment()
            s2.points[0] = self.vector2ground(s1.pixels_normalized[0])
            s2.points[1] = self.vector2ground(s1.pixels_normalized[1])
            s2.pixels_normalized[0] = s1.pixels_normalized[0]
            s2.pixels_normalized[1] = s1.pixels_normalized[1] 
            s2.color = s1.color
            # TODO what about normal and points
            sl2.segments.append(s2)
        return sl2


def load_board_info(filename=None):
    '''Load calibration checkerboard info'''
    if filename is None:
        root = dtu.get_ros_package_path('duckietown')
        filename = root + '/config/baseline/ground_projection/ground_projection/default.yaml'
        
    if not os.path.isfile(filename):
        msg = 'No such file: %s' % filename
        raise dtu.DTException(msg)
        
    target_data = dtu.yaml_load_file(filename)
    target_info = {
        'width': target_data['board_w'],
        'height': target_data['board_h'],
        'square_size': target_data['square_size'],
        'x_offset': target_data['x_offset'],
        'y_offset': target_data['y_offset'],
        'offset': np.array([target_data['x_offset'], -target_data['y_offset']]),
        'size': (target_data['board_w'], target_data['board_h']),
      }
    dtu.logger.info("Loaded checkerboard parameters")
    return target_info

