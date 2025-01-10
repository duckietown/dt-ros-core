import time
import numpy as np
import cv2
from BaseComNode import BaseComNode
from io import BytesIO

TEST_VIDEO = '/home/sh3mm/Desktop/IFT6757/image/videos/video-Duckiebot-2Hz-10Hz.avi'
#TEST_VIDEO = '/home/alaeddine/1IFT6757/Project/VideoCaptures/video-TL.avi'
np.set_printoptions(linewidth=np.inf)


class Data:
    def __init__(self, data):
        self.data = data


def main():
    vid = cv2.VideoCapture(TEST_VIDEO)

    node = BaseComNode()

    node.intersection_type_callback(1)

    while vid.isOpened():
        ret, curr_img = vid.read()
        if not ret:
            break

        # data prep to simulate the actual input
        _, buffer = cv2.imencode(".jpeg", curr_img)
        io_buf = BytesIO(buffer)
        data = Data(io_buf.getbuffer())

        # node call
        node.img_callback(data)
        time.sleep(1/30)

        node.run()


if __name__ == '__main__':
    main()
