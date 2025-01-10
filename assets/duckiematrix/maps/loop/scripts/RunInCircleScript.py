from duckiematrix_engine.template import MatrixEntityBehavior

import copy
import numpy as np


class RunInCircleScript(MatrixEntityBehavior):

    def __init__(self, *args, radius: float = 0.2, speed: float = 0.5):
        super(RunInCircleScript, self).__init__(*args)
        self._initial_pose = copy.deepcopy(self.pose.as_frame()['pose'])
        self._radius = radius
        self._speed = speed
        self._time = 0

    def update(self, delta_t: float):
        self._time += delta_t
        self.pose.x = self._initial_pose['x'] + self._radius * np.sin(self._time * self._speed)
        self.pose.y = self._initial_pose['y'] + self._radius * np.cos(self._time * self._speed)
        self.pose.yaw = -(self._time * self._speed) + np.deg2rad(180)
        # update frames
        self.pose.commit()
