from abc import ABCMeta, abstractmethod
from collections import namedtuple

import duckietown_code_utils as dtu


# TODO[ente-newdeal]: this has been moved to `dt_computer_vision.line_detector`


FAMILY_LINE_DETECTOR = "line_detector"

Detections = namedtuple("Detections", ["lines", "normals", "area", "centers"])

__all__ = ["LineDetectorInterface", "Detections", "FAMILY_LINE_DETECTOR"]


class LineDetectorInterface(metaclass=ABCMeta):
    @abstractmethod
    def setImage(self, bgr: dtu.NPImageBGR):
        pass

    @abstractmethod
    def detectLines(self, color):
        """Returns a tuple of class Detections"""
