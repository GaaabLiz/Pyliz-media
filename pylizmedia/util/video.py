import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Tuple

import numpy as np
import cv2

import ffmpeg

from pylizmedia.domain.video import FrameOptions, Frame, SceneType
from pylizmedia.log.pylizLogger import logger
from pylizlib.os.pathutils import check_path_file, check_path


class FrameSelector(ABC):
    """Abstract base class for frame selection strategies"""

    def __init__(self):
        self.logger = logger

    @abstractmethod
    def select_frames(self, video_path: str, frame_options: FrameOptions) -> List[Frame]:
        pass

    def _validate_video(self, video_path: str) -> Tuple[cv2.VideoCapture, float, float, int]:
        """Validate video file and return video properties"""
        self.logger.trace(f"Validating video file: {video_path}")
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps

        self.logger.trace(f"Video properties - FPS: {fps}, Total frames: {total_frames}, Duration: {duration:.2f}s")
        return cap, fps, duration, total_frames


class DynamicFrameSelector(FrameSelector):
    """Dynamic frame selection strategy based on scene changes and motion"""

    def select_frames(self, video_path: str, frame_options: FrameOptions) -> List[Frame]:
        self.logger.trace("Starting dynamic frame selection")
        cap, fps, duration, total_frames = self._validate_video(video_path)

        scene_changes = self._detect_scene_changes(video_path, cap)
        target_frames = frame_options.calculate_dynamic_frame_count(duration, scene_changes)

        self.logger.trace(f"Target frames for analysis: {target_frames}")
        frames = self._extract_frames(cap, target_frames, scene_changes)

        cap.release()
        self.logger.trace(f"Dynamic frame selection completed. Selected {len(frames)} frames")
        return frames

    def _detect_scene_changes(self, video_path: str, cap: cv2.VideoCapture, threshold: float = 20.0) -> List[float]:
        """Detect significant scene changes in the video"""
        self.logger.trace("Detecting scene changes")
        scene_changes = []
        prev_frame = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            if prev_frame is not None:
                diff_score = self._calculate_frame_difference(prev_frame, frame)

                if diff_score > threshold:
                    timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                    scene_changes.append(timestamp)

            prev_frame = frame

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return scene_changes

    def _calculate_frame_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate the difference between two frames"""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        diff = cv2.absdiff(gray1, gray2)
        return np.mean(diff)

    def _extract_frames(self, cap: cv2.VideoCapture, target_frames: int, scene_changes: List[float]) -> List[Frame]:
        """Extract frames based on scene changes and target count"""
        frames = []
        ret, first_frame = cap.read()
        if ret:
            frames.append(Frame(
                image=cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB),
                timestamp=0.0,
                scene_type=SceneType.STATIC
            ))

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        interval = total_frames // (target_frames - 1)

        for frame_idx in range(interval, total_frames, interval):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            is_scene_change = any(abs(sc - timestamp) < 0.1 for sc in scene_changes)
            scene_type = SceneType.TRANSITION if is_scene_change else SceneType.STATIC

            frames.append(Frame(
                image=frame_rgb,
                timestamp=timestamp,
                scene_type=scene_type
            ))

        return frames


class UniformFrameSelector(FrameSelector):
    """Uniform frame selection strategy selecting frames at regular intervals"""

    def select_frames(self, video_path: str, frame_options: FrameOptions) -> List[Frame]:
        self.logger.trace("Starting uniform frame selection")
        cap, fps, duration, total_frames = self._validate_video(video_path)

        target_frames = frame_options.calculate_uniform_frame_count(duration)
        self.logger.trace(f"Target frames for uniform selection: {target_frames}")
        frames = self._extract_uniform_frames(cap, target_frames, fps)

        cap.release()
        self.logger.trace(f"Uniform frame selection completed. Selected {len(frames)} frames")
        return frames

    def _extract_uniform_frames(self, cap: cv2.VideoCapture, target_frames: int, fps: float) -> List[Frame]:
        """Extract frames at uniform intervals"""
        frames = []
        if target_frames <= 0:
            return frames

        interval = cap.get(cv2.CAP_PROP_FRAME_COUNT) / target_frames
        self.logger.debug(f"Frame extraction interval: {interval}")

        for i in range(target_frames):
            frame_number = int(i * interval)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            ret, frame = cap.read()
            if not ret:
                self.logger.warning(f"Failed to read frame at position {frame_number}")
                continue

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frames.append(Frame(
                image=frame_rgb,
                timestamp=timestamp,
                scene_type=SceneType.STATIC
            ))

        return frames


class AllFrameSelector(FrameSelector):
    """Frame selection strategy that selects all frames from the video"""

    def select_frames(self, video_path: str, frame_options: FrameOptions) -> List[Frame]:
        self.logger.trace("Starting all frame selection")
        cap, fps, duration, total_frames = self._validate_video(video_path)

        frames = self._extract_all_frames(cap, fps)
        cap.release()
        self.logger.trace(f"All frame selection completed. Selected {len(frames)} frames")
        return frames

    def _extract_all_frames(self, cap: cv2.VideoCapture, fps: float) -> List[Frame]:
        """Extract all frames from the video"""
        frames = []
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frames.append(Frame(
                image=frame_rgb,
                timestamp=timestamp,
                scene_type=SceneType.STATIC  # Or determine dynamically if needed
            ))

            frame_idx += 1
            if frame_idx % 100 == 0:
                self.logger.debug(f"Extracted {frame_idx} frames")

        return frames


class VideoUtils:

    @staticmethod
    def extract_audio(video_path, audio_path, use_existing=False):
        if use_existing and os.path.exists(audio_path):
            logger.debug(f"Audio file for {Path(video_path).name} already exist: {audio_path}")
            return
        ffmpeg.input(video_path).output(audio_path).run(overwrite_output=True)

    # @staticmethod
    # def _extract_audio_librosa(video_path: str, target_sampling_rate) -> Tuple[np.ndarray, int]:
    #     """Extract audio from video file and return as numpy array with sampling rate using librosa"""
    #     try:
    #         # Load audio using librosa
    #         raw_audio, original_sampling_rate = librosa.load(
    #             video_path,
    #             sr=target_sampling_rate,
    #             mono=True
    #         )
    #
    #         # Ensure float32 dtype and normalize
    #         raw_audio = raw_audio.astype(np.float32)
    #         if np.abs(raw_audio).max() > 1.0:
    #             raw_audio = raw_audio / np.abs(raw_audio).max()
    #
    #         logger.debug(f"Raw audio shape: {raw_audio.shape}, dtype: {raw_audio.dtype}")
    #
    #         return raw_audio, original_sampling_rate
    #
    #     except Exception as e:
    #         logger.error(f"Error extracting audio with librosa: {str(e)}")
    #         raise


    @staticmethod
    def extract_frame_advanced(
            video_path,
            frame_folder,
            frame_selector: FrameSelector,
            frame_options: FrameOptions = FrameOptions(),
            use_existing=True
    ):
        # # controllo se esistono già i frame
        # if use_existing and len(os.listdir(frame_folder)) > 0:
        #     logger.debug(f"Frames already exist in {frame_folder}. Reading existing frames...")
        #     frames = imgutils.load_images_as_ndarrays(frame_folder)
        #     if len(frames) > 0:
        #         return frames
        #     else:
        #         logger.warning(f"Frames has been found in {frame_folder}, but no frames were loaded. Extracting frames again...")
        #         shutil.rmtree(frame_folder)

        # estratto i frame con il frame selector
        frame_list = frame_selector.select_frames(video_path, frame_options)
        frames_list_images = [frame.image for frame in frame_list]
        imgutils.save_ndarrays_as_images(frames_list_images, frame_folder)
        return frame_list


    @staticmethod
    def extract_frames_thr(
            video_path,
            output_folder,
            difference_threshold=30,
            use_existing=True
    ):
        pathutils.check_path_file(video_path)
        pathutils.check_path(output_folder, True)

        # Se esistono già i frame, non fare nulla
        if use_existing and len(os.listdir(output_folder)) > 0:
            logger.debug(f"Frames already exist in {output_folder}. Exiting frame extraction.")
            return

        # Apri il video
        cap = cv2.VideoCapture(video_path)

        # Contatore per numerare i frame
        frame_count = 0
        saved_frame_count = 0

        # Leggi il primo frame
        ret, prev_frame = cap.read()
        if not ret:
            cap.release()
            cv2.destroyAllWindows()
            raise Exception("OPenCV error: Error reading video")

        # Converti il primo frame in scala di grigi
        prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

        # Salva il primo frame
        file_name = os.path.basename(video_path).split(".")[0]
        frame_path = os.path.join(output_folder, f"{file_name}_frame_{saved_frame_count}.jpg")
        cv2.imwrite(frame_path, prev_frame)
        saved_frame_count += 1

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Converti il frame corrente in scala di grigi
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Calcola la differenza assoluta media tra il frame corrente e quello precedente
            diff = cv2.absdiff(frame_gray, prev_frame_gray)
            mean_diff = np.mean(diff)

            # Se la differenza è maggiore della soglia, salva il frame
            if mean_diff > difference_threshold:
                file_name = os.path.basename(video_path).split(".")[0]
                frame_path = os.path.join(output_folder, f"{file_name}_frame_{saved_frame_count}.jpg")
                cv2.imwrite(frame_path, frame)
                saved_frame_count += 1
                prev_frame_gray = frame_gray  # Aggiorna il frame precedente
                logger.trace(f"Frame {frame_count} saved because threshold exceeded: {mean_diff}")

            frame_count += 1
            logger.trace(f"Frame {frame_count} processed, {saved_frame_count} frames saved")

        # Rilascia la cattura del video e chiudi le finestre
        cap.release()
        cv2.destroyAllWindows()
