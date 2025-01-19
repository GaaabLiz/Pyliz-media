
import unittest


import sys
import os

from pylizlib.os import pathutils

from pylizmedia.util.vidutils import VideoUtils
from pylizmedia.video.FrameSelectors import DynamicFrameSelector, UniformFrameSelector


class TestVideo(unittest.TestCase):

    def testFrames(self):
        path = "/Users/gabliz/Movies/marco.mp4"
        frame_folder = "/Users/gabliz/.pyliz/temp/frame"
        VideoUtils.extract_frames_thr(path, frame_folder, 80)

    def testFramesAdv(self):
        path = "/Users/gabliz/Movies/marco.mp4"
        frame_folder = "/Users/gabliz/.pyliz/temp/frame"
        pathutils.check_path(frame_folder, True)
        VideoUtils.extract_frame_advanced(path, frame_folder, DynamicFrameSelector())


