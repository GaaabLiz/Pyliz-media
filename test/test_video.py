
import unittest


import sys
import os

import pylizlib.log.pylizLogging
from loguru import logger
from pylizlib.os import pathutils

from pylizmedia.log.pylizMediaLogging import LOGGER_PYLIZ_MEDIA_NAME
from pylizmedia.util.vidutils import VideoUtils
from pylizmedia.video.FrameSelectors import DynamicFrameSelector, UniformFrameSelector


class TestVideo(unittest.TestCase):

    def testFrames(self):
        path = "/Users/gabliz/Movies/marco.mp4"
        frame_folder = "/Users/gabliz/.pyliz/temp/frame"
        logger.enable(LOGGER_PYLIZ_MEDIA_NAME)
        VideoUtils.extract_frames_thr(path, frame_folder, 80)

    def testFramesAdv(self):
        path = "/Users/gabliz/Movies/marco.mp4"
        frame_folder = "/Users/gabliz/.pyliz/temp/frame"
        pathutils.check_path(frame_folder, True)
        VideoUtils.extract_frame_advanced(path, frame_folder, DynamicFrameSelector())


