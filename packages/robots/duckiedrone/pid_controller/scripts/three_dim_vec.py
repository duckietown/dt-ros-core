from dataclasses import astuple, dataclass
from geometry_msgs.msg import Vector3

import numpy as np


@dataclass
class ThreeDimVec:
    """ Struct to store three variables referenced as x,y,z """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    @property
    def list(self):
        return list(astuple(self))

    def as_ros_vector3(self):
        return Vector3(x=self.x, y=self.y, z=self.z)

    def __iter__(self):
        return iter((self.x, self.y, self.z))  # Return an iterator over the fields

    def __str__(self):
        return "x: %f, y: %f, z: %f" % (self.x, self.y, self.z)

    def __mul__(self, other):
        return ThreeDimVec(self.x * other, self.y * other, self.z * other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        return ThreeDimVec(self.x / other, self.y / other, self.z / other)

    def __add__(self, other):
        return ThreeDimVec(self.x + other.x, self.y + other.y, self.z + other.z)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return ThreeDimVec(self.x - other.x, self.y - other.y, self.z - other.z)

    @property
    def magnitude(self):
        return np.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    @property
    def xy_magnitude(self):
        return np.sqrt(self.x * self.x + self.y * self.y)


@dataclass
class Position(ThreeDimVec):
    """ Struct to store position components x,y,z """
    pass

class Velocity(ThreeDimVec):
    """ Struct to store velocity components x,y,z"""
    pass

class Error(ThreeDimVec):
    """ Struct to store 3D errors which is in the form x,y,z"""
    pass

class RPY(ThreeDimVec):
    """ Struct to store the roll, pitch, in x,y,z"""

    def __init__(self, r=0.0, p=0.0, y=0.0):
        super(RPY, self).__init__(r, p, y)
        self.r = self.x
        self.p = self.y
        self.y = self.z