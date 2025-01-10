from typing import Tuple, List, Iterable, Union
from itertools import chain
import scipy.ndimage.filters as filters
import numpy as np
import cv2


class Point:
    """
    Class used to encapsulate point related methods and variables.

    This class is used to track the flashing rate of a point in an image stream using the positive component of the
    second derivative (the acceleration of intensity change). This is used to make sure the change is causes by a
    flashing and not the autofocus of the camera.
    """
    # Radius used to crop image around the point when making the histogram
    RADIUS = 10

    def __init__(self, new_coords: Tuple[int, int], last_seen: int = 0, weight: int = 1):
        self.coords = new_coords  # coordinates of the point in the image
        self.last_seen = last_seen  # number of frames since activity was register at the point
        self.weight = weight  # how many points are approximated by this one
        self.prev_frames: List[Union[None, int]] = [None, None]  # image intensity buffer
        self.histogram = np.array([])  # histogram of the intensity acceleration

    def __add__(self, other: 'Point'):
        """
        This overload of the add operator is created to facilitate the point merger.
        It also makes using python's sum function possible.
        """
        # the new weight is a sum of the two components since it represent how many points are being approximated
        new_weight = other.weight + self.weight
        # the new coords are a weighted average of both points rounded down for stability
        new_coords = (
            round((other.coords[0] * other.weight + self.coords[0] * self.weight) / new_weight),
            round((other.coords[1] * other.weight + self.coords[1] * self.weight) / new_weight)
        )
        # the last_seen variable is the min of both values since both points are now the same
        new_last_seen = min(other.last_seen, self.last_seen)

        new_point = Point(new_coords, new_last_seen, new_weight)
        # the histogram related variables are taken from the point with the biggest weight since it means more stability
        priority_point = [self, other][self.weight < other.weight]
        new_point.prev_frames = priority_point.prev_frames
        new_point.histogram = priority_point.histogram
        return new_point
    
    def __repr__(self):
        return f"coords: {self.coords}, last_seen: {self.last_seen}"#, hist: {self.histogram}"

    def add_frame(self, frame: np.ndarray):
        """
        This function gives the point a new full frame to add to its histogram. has mentioned in the class description,
        it does so by calculating the second derivative of the frame using the last two.

        :param frame: an image in the form of a ndarray
        """
        # first, get the intensity of the frame around the point
        frame = np.sum(self.get_sub(frame))

        # if the point is at least two frame old before now, append the second derivative to the histogram
        if None not in self.prev_frames:
            delta = [
                np.clip(self.prev_frames[1] - self.prev_frames[0], 0, np.inf),
                np.clip(frame - self.prev_frames[0], 0, np.inf)
            ]
            self.histogram = np.append(self.histogram, np.clip(delta[1] - delta[0], 0, np.inf))

        # update the cache to keep the new frame intensity value and remove the stale one
        self.prev_frames[1] = self.prev_frames[0]
        self.prev_frames[0] = int(np.sum(frame, dtype=int))

    def get_frequency(self, frame_rate=30, tolerance=.30) -> Tuple[float, np.ndarray]:
        """
        This method calculates the flashing frequency of a point by looking  at spikes in the histogram of intensity
        change acceleration and taking the median of the time between spikes.

        :param frame_rate: the framerate of the image stream
        :param tolerance: what is the minimum percentage of the max value should be used as a threshold
                          for a spike identification
        :return: the calculated frequency and the histogram of spikes as an array of ones and zeroes. ex: [1,0,0,0,1,0]
        """
        # if the sum of the whole histogram is too small, probably a false point and should not have a framerate.
        if not np.sum(self.histogram > 5e3):
            return -1, np.zeros(self.histogram.shape).astype(int)

        spikes = (self.histogram / np.max(self.histogram)) > tolerance

        # if there is less than 2 spikes, nothing can be inferred
        if np.sum(spikes) < 2:
            return -1, spikes.astype(int)

        # A frequency of 30 Hz is impossible, so two adjacent spikes are impossible
        # and mostly mean that the spike happened between two frames
        for i, v in enumerate(spikes[1:]):
            if v and spikes[i]:
                spikes[i] = False

        spike_freq = frame_rate / np.median(np.diff(spikes.nonzero()))
        return spike_freq, spikes.astype(int)

    @staticmethod
    def batch(batch: Iterable[Tuple[int, int]]) -> List['Point']:
        """
        Creates a batch of points from x, y coords tuple.

        :param batch: an Iterable that returns coords tuple
        :return: a list of points
        """
        return [Point(b) for b in batch]

    @staticmethod
    def group_point(points: List['Point'], size) -> List['Point']:
        """
        This method groups points according to their proximity to one another.

        :param points: the list of points to be grouped
        :param size: the range at which a point can be merged into another
        :return: A List of points smaller or equal to the one given in inputs
        """
        group_nb = 0
        groups = []
        for p1 in points:
            i1, j1 = p1.coords
            for p2, nb in chain(*groups):
                i2, j2 = p2.coords
                # if a point is close enough to a point already grouped, add it to that group
                if abs(i2 - i1) < size and abs(j2 - j1) < size:
                    groups[nb].append((p1, nb))
                    break
            # if the point is too far from any other already grouped
            # or if there are no groups yet, add the point to a new group
            else:
                groups.append([(p1, group_nb)])
                group_nb += 1

        # this fuses all the points of each group into a single point
        # such that there is only a single point left per group
        return [sum((e[0] for e in group[1:]), start=group[0][0]) for group in groups]

    def get_sub(self, img_buffer: Iterable[np.ndarray]) -> np.ndarray:
        """
        This method simply returns the sub image around the point using a predefined radius
        """
        return np.array(img_buffer)[
           int(self.coords[0] - self.RADIUS): int(self.coords[0] + self.RADIUS + 1),
           int(self.coords[1] - self.RADIUS): int(self.coords[1] + self.RADIUS + 1)
        ]


def get_maxes(base_img: np.ndarray, size=9, threshold=120, sigmaX=3, sigmaY=3):
    """
    This function returns a group of points representing the local maximums present in an image. This is used to get
    the points where the rate of change is the highest (potential LEDs in the context of the car).

    :param base_img: the image to be analysed in the form of a ndarray
    :param threshold: the minimum brightness value to be considered for it to be a maximum
    :param size: minimum distance between points
    :param sigmaX: Gaussian blur kernel standard deviation in X direction.
    :param sigmaY: Gaussian blur kernel standard deviation in Y direction.
    :return: a List of Points
    """
    gaussian_blur = cv2.GaussianBlur(src=np.uint8(base_img), ksize=(size, size), sigmaX=sigmaX, sigmaY=sigmaY)
    data_max = filters.maximum_filter(gaussian_blur, size)

    maxima = (gaussian_blur == data_max)

    diff = ((data_max - 0) > threshold)
    maxima[diff == 0] = 0

    return Point.group_point(Point.batch(zip(*maxima.nonzero())), 2)
