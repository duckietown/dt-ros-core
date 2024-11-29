"""
slam_helper.py

Implements fastSLAM for the pidrone
"""

import numpy as np
import math
import slam_helper.utils as utils
import copy
import cv2
import threading

# ----- camera parameters ----- #
CAMERA_SCALE = 290.0
CAMERA_WIDTH = 480.0
CAMERA_HEIGHT = 640.0
MATCH_RATIO = 0.7

# ----- SLAM parameters ----- #
PROB_THRESHOLD = 0.005
KEYFRAME_DIST_THRESHOLD = CAMERA_HEIGHT
KEYFRAME_YAW_THRESHOLD = 0.175

# Optional pose/weight logging
POSE = False
WEIGHT = False
pose_path = '/home/luke/ws/src/pidrone_pkg/scripts/pose_data.txt'


class Particle:
    """
    Represents a particle in FastSLAM, containing the robot's pose, landmarks, and weight.
    """

    def __init__(self, x, y, z, yaw):
        self.pose = [x, y, z, yaw]
        self.landmarks = []
        self.weight = PROB_THRESHOLD

    def __str__(self):
        return f"Pose: {self.pose} Weight: {self.weight}"


class FastSLAM:
    """
    Implements the FastSLAM algorithm for pose estimation and mapping.
    """

    def __init__(self):
        self.particles = None
        self.num_particles = None
        self.weight = PROB_THRESHOLD
        self.z = 0
        self.perceptual_range = 0.0

        self.key_kp = None
        self.key_des = None

        if POSE or WEIGHT:
            self.file = open(pose_path, 'w')

        # Threading for parallel updates
        self.thread_queue = utils.ThreadQueue()
        self.most_recent_map = None
        self.new_result = False

        # OpenCV feature matching parameters
        index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)

        # Noise for observations
        self.sigma_d = 3
        self.sigma_p = 0.30
        self.sigma_observation = np.array([[self.sigma_d ** 2, 0], [0, self.sigma_p ** 2]])

        # Noise for motion updates
        sigma_vx, sigma_vy, sigma_vz, sigma_yaw = 2, 2, 0.0, 0.01
        self.covariance_motion = np.array([
            [sigma_vx ** 2, 0, 0, 0],
            [0, sigma_vy ** 2, 0, 0],
            [0, 0, sigma_vz ** 2, 0],
            [0, 0, 0, sigma_yaw ** 2]
        ])

    def generate_particles(self, num_particles):
        """
        Creates the initial set of particles for FastSLAM.
        Each particle starts near (0, 0) with some noise.

        :param num_particles: Number of particles to generate
        :return: Initial estimated pose
        """
        self.particles = [
            Particle(abs(utils.normal(0, 0.1)), abs(utils.normal(0, 0.1)), self.z, abs(utils.normal(math.pi, 0.01)))
            for _ in range(num_particles)
        ]

        self.num_particles = num_particles
        self.key_kp, self.key_des, self.most_recent_map = None, None, None
        self.new_result = False
        self.weight = PROB_THRESHOLD

        return self.estimate_pose()

    def run(self, z, prev_kp, prev_des, kp, des):
        """
        Executes one iteration of FastSLAM, including motion prediction and map update.

        :param z: New infrared height estimate for the drone
        :param prev_kp: Previous keypoints
        :param prev_des: Previous descriptors
        :param kp: Current keypoints
        :param des: Current descriptors
        :return: Updated pose estimate and weight
        """
        self.z = z
        self.update_perceptual_range()

        transform = utils.compute_transform(self.matcher, prev_kp, prev_des, kp, des)

        if transform is not None:
            x = -transform[0, 2]
            y = transform[1, 2]
            yaw = -np.arctan2(transform[1, 0], transform[0, 0])

            # Predict particle positions
            for particle in self.particles:
                self.predict_particle(particle, x, y, yaw)

            # Map update based on keyframes
            self.detect_keyframe(kp, des)

            if self.new_result:
                self.new_result = False
                self.update_particles_from_map()

        return self.estimate_pose(), self.weight

    def predict_particle(self, particle, x, y, yaw):
        """
        Updates a particle's pose based on motion prediction.

        :param particle: Particle to update
        :param x: Control input x
        :param y: Control input y
        :param yaw: Control input yaw
        """
        noisy_x_y_z_yaw = np.random.multivariate_normal([x, y, self.z, yaw], self.covariance_motion)

        particle.pose[0] += self.pixel_to_meter(noisy_x_y_z_yaw[0])
        particle.pose[1] += self.pixel_to_meter(noisy_x_y_z_yaw[1])
        particle.pose[2] = self.z
        particle.pose[3] += noisy_x_y_z_yaw[3]
        particle.pose[3] = utils.adjust_angle(particle.pose[3])

    def detect_keyframe(self, kp, des):
        """
        Detects whether a new keyframe should be set, based on the current motion.

        :param kp: Current keypoints
        :param des: Current descriptors
        """
        if self.key_kp is not None and self.key_des is not None:
            transform = utils.compute_transform(self.matcher, self.key_kp, self.key_des, kp, des)

            if transform is not None:
                x = self.pixel_to_meter(-transform[0, 2])
                y = self.pixel_to_meter(transform[1, 2])
                yaw = -np.arctan2(transform[1, 0], transform[0, 0])

                if utils.distance(x, y, 0, 0) > self.pixel_to_meter(KEYFRAME_DIST_THRESHOLD) or yaw > KEYFRAME_YAW_THRESHOLD:
                    self.start_map_update_thread(kp, des)
            else:
                self.start_map_update_thread(kp, des)
        else:
            self.start_map_update_thread(kp, des)

    def start_map_update_thread(self, kp, des):
        """
        Starts a thread for updating the map using the current keypoints and descriptors.

        :param kp: Current keypoints
        :param des: Current descriptors
        """
        t = threading.Thread(target=self.update_map, args=(kp, des,))
        self.thread_queue.add_thread(t)
        self.key_kp, self.key_des = kp, des

    def update_map(self, kp, des):
        """
        Updates the particle map in a separate thread.

        :param kp: Current keypoints
        :param des: Current descriptors
        """
        curr_particles = copy.deepcopy(self.particles)

        for particle in curr_particles:
            self.update_particle(particle, kp, des)

        self.most_recent_map = curr_particles
        self.new_result = True

    def update_particle(self, particle, keypoints, descriptors):
        """
        Updates a particle's landmarks based on observed keypoints and descriptors.

        :param particle: Particle to update
        :param keypoints: Observed keypoints
        :param descriptors: Observed descriptors
        """
        particle.weight = PROB_THRESHOLD

        if len(particle.landmarks) == 0:
            for kp, des in zip(keypoints, descriptors):
                utils.add_landmark(particle, kp, des, self.sigma_observation, self.kp_to_measurement)
                particle.weight += math.log(PROB_THRESHOLD)
        else:
            close_landmarks = self.get_close_landmarks(particle)

            part_descriptors = [lm[0].des for lm in close_landmarks] if close_landmarks else None
            matched_landmarks = [False] * len(close_landmarks) if close_landmarks else None

            for kp, des in zip(keypoints, descriptors):
                match = self.matcher.knnMatch(np.array([des]), np.array(part_descriptors), k=2) if part_descriptors else None

                if not self.__is_valid_match(match):
                    utils.add_landmark(particle, kp, des, self.sigma_observation, self.kp_to_measurement)
                    particle.weight += math.log(PROB_THRESHOLD)
                else:
                    close_index = match[0].trainIdx
                    matched_landmarks[close_index] = True
                    lm = close_landmarks[close_index][0]

                    updated_landmark = utils.update_landmark(particle, lm, kp, des, self.sigma_observation, self.kp_to_measurement)
                    particle.landmarks[close_landmarks[close_index][1]] = updated_landmark
                    particle.weight += math.log(self.scale_weight(match[0].distance, match[1].distance))

            if matched_landmarks is not None:
                self.update_landmark_counters(particle, close_landmarks, matched_landmarks)

    def get_close_landmarks(self, particle):
        """
        Retrieves landmarks that are within the perceptual range of a given particle.

        :param particle: Particle whose landmarks are to be checked
        :return: List of close landmarks
        """
        close_landmarks = []
        for i, lm in enumerate(particle.landmarks):
            if utils.distance(lm.x, lm.y, particle.pose[0], particle.pose[1]) <= self.perceptual_range * 1.2:
                close_landmarks.append((lm, i))
        return close_landmarks

    def __is_valid_match(self, match):
        """
        Checks if a feature match is valid based on the distance ratio.

        :param match: List of matches
        :return: True if the match is valid, otherwise False
        """
        return match is not None and len(match) >= 2 and match[0].distance < MATCH_RATIO * match[1].distance

    def update_landmark_counters(self, particle, close_landmarks, matched_landmarks):
        """
        Updates the counters of landmarks based on whether they were matched or not.

        :param particle: Particle being updated
        :param close_landmarks: List of close landmarks
        :param matched_landmarks: Boolean list indicating whether each landmark was matched
        """
        removed_landmarks = []
        for i, match in enumerate(matched_landmarks):
            lm, index = close_landmarks[i]
            if match:
                lm.counter += 1
            else:
                lm.counter -= 1
                particle.weight += math.log(0.1 * PROB_THRESHOLD)
                if lm.counter < 0:
                    removed_landmarks.append(lm)

        for lm in removed_landmarks:
            particle.landmarks.remove(lm)

    def update_particles_from_map(self):
        """
        Updates the current particle set with the most recent map generated by the background thread.
        """
        most_recent_particles = self.most_recent_map
        for old_particle, new_particle in zip(self.particles, most_recent_particles):
            old_particle.landmarks = new_particle.landmarks
            old_particle.weight = new_particle.weight

        if WEIGHT:
            self.file.write(str([p.weight for p in self.particles]) + '\n')

        self.weight = self.get_average_weight()
        self.resample_particles()

    def resample_particles(self):
        """
        Resamples particles according to their weight to focus on the best particles.
        """
        weights = [p.weight for p in self.particles]
        lowest_weight = min(weights)

        normal_weights = np.array([1 - (w / lowest_weight) if w != 0 else PROB_THRESHOLD for w in weights])
        normal_weights /= np.sum(normal_weights)

        samples = np.random.multinomial(self.num_particles, normal_weights)
        new_particles = [copy.deepcopy(self.particles[i]) for i, count in enumerate(samples) for _ in range(count)]

        self.particles = new_particles

    def get_average_weight(self):
        """
        Calculates the average weight of the particles.

        :return: Average weight of particles
        """
        return np.sum([p.weight for p in self.particles]) / float(self.num_particles)

    def pixel_to_meter(self, px):
        """
        Converts pixel distances into meters using the camera's scale.

        :param px: Pixel distance
        :return: Equivalent distance in meters
        """
        return px * self.z / CAMERA_SCALE

    def estimate_pose(self):
        """
        Estimates the current pose of the robot by averaging the poses of all particles.

        :return: Estimated pose [x, y, z, yaw]
        """
        weights = [p.weight for p in self.particles]
        lowest_weight = min(weights)

        normal_weights = np.array([1 - (w / lowest_weight) if w != 0 else PROB_THRESHOLD for w in weights])
        normal_weights /= np.sum(normal_weights)

        x, y, z, yaw = 0.0, 0.0, 0.0, 0.0
        for i, prob in enumerate(normal_weights):
            x += prob * self.particles[i].pose[0]
            y += prob * self.particles[i].pose[1]
            z += prob * self.particles[i].pose[2]
            yaw += prob * self.particles[i].pose[3]

        return [x, y, z, utils.adjust_angle(yaw)]

    def update_perceptual_range(self):
        """
        Updates the perceptual range of the robot, which defines the distance at which landmarks can be observed.
        """
        self.perceptual_range = self.pixel_to_meter(CAMERA_WIDTH / 2)

    def kp_to_measurement(self, kp):
        """
        Converts a keypoint into a range and bearing measurement relative to the camera center.

        :param kp: Keypoint to be converted
        :return: Range and bearing to the keypoint
        """
        kp_x, kp_y = kp.pt[0], kp.pt[1]
        kp_y = CAMERA_HEIGHT - kp_y

        dx = kp_x - CAMERA_WIDTH / 2
        dy = kp_y - CAMERA_HEIGHT / 2
        dist = math.sqrt(dx ** 2 + dy ** 2)
        dist = self.pixel_to_meter(dist)
        bearing = math.atan2(dy, dx)

        return dist, bearing

    def scale_weight(self, match0, match1):
        """
        Scales the weight of a particle based on the distances of the two best feature matches.

        :param match0: Distance of the best match
        :param match1: Distance of the second-best match
        :return: Scaled weight between 0 and 1
        """
        scaled = (match1 - match0) / float(match1)
        return scaled if scaled != 0 else PROB_THRESHOLD
