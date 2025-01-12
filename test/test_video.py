
import unittest


import sys
import os

from pylizmedia.util.vidutils import VideoUtils


class TestVideo(unittest.TestCase):

    def testFrames(self):
        path = "/Users/gabliz/Movies/video6.mov"
        VideoUtils.extract_frames(path, "/Users/gabliz/Developer/Pyliz-media/temp", 80)


