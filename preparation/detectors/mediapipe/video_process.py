#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2023 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import cv2
import numpy as np


def linear_interpolate(landmarks, start_idx, stop_idx):
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx - start_idx):
        landmarks[start_idx + idx] = (
            start_landmarks + idx / float(stop_idx - start_idx) * delta
        )
    return landmarks


class VideoProcess:
    def __init__(
        self,
        crop_width=96,
        crop_height=96,
        start_idx=3,
        stop_idx=4,
        window_margin=12,
    ):
        self.crop_width = crop_width
        self.crop_height = crop_height
        self.start_idx = start_idx
        self.stop_idx = stop_idx
        self.window_margin = window_margin

    def __call__(self, video, landmarks):
        # Pre-process landmarks: interpolate frames that are not detected
        preprocessed_landmarks = self.interpolate_landmarks(landmarks)
        # Exclude corner cases: no landmark in all frames
        if not preprocessed_landmarks:
            return
        # Affine transformation and crop patch
        sequence = self.crop_patch(video, preprocessed_landmarks)
        assert sequence is not None, "crop an empty patch."
        return sequence

    def crop_patch(self, video, landmarks):
        sequence = []
        for frame_idx, frame in enumerate(video):
            new_frame = np.zeros((self.crop_width, self.crop_height))
            x_coords, y_coords = [point[0] for point in landmarks[frame_idx]], [point[1] for point in landmarks[frame_idx]]
            x_min, x_max, y_min, y_max = min(x_coords), max(x_coords), min(y_coords), max(y_coords)
            x_c, y_c = (x_min+x_max) // 2, (y_min+y_max) // 2

            # fitting landmarks into 96x96 (drawing them with radius 1 and having a margin
            lip_reg_width = self.crop_height - self.window_margin
            scaling_coef = min([lip_reg_width/(x_c-x_min), lip_reg_width/(x_max-x_c), lip_reg_width/(y_c-y_min), lip_reg_width/(y_max-y_c)])/2
            x_resized = (x_coords - x_c) * scaling_coef + lip_reg_width//2 + self.window_margin//2
            y_resized = (y_coords - y_c) * scaling_coef + lip_reg_width//2 + self.window_margin//2
            resized_projections = list(zip(x_resized,y_resized))

            for proj in resized_projections:
                new_frame = cv2.circle(new_frame, (int(proj[0]),int(proj[1])), radius=1, color=255, thickness=-1)
            sequence.append(new_frame)
        return np.array(sequence)

    def interpolate_landmarks(self, landmarks):
        valid_frames_idx = [idx for idx, lm in enumerate(landmarks) if lm is not None]

        if not valid_frames_idx:
            return None

        for idx in range(1, len(valid_frames_idx)):
            if valid_frames_idx[idx] - valid_frames_idx[idx - 1] > 1:
                landmarks = linear_interpolate(
                    landmarks, valid_frames_idx[idx - 1], valid_frames_idx[idx]
                )

        valid_frames_idx = [idx for idx, lm in enumerate(landmarks) if lm is not None]

        # Handle corner case: keep frames at the beginning or at the end that failed to be detected
        if valid_frames_idx:
            landmarks[: valid_frames_idx[0]] = [
                landmarks[valid_frames_idx[0]]
            ] * valid_frames_idx[0]
            landmarks[valid_frames_idx[-1] :] = [landmarks[valid_frames_idx[-1]]] * (
                len(landmarks) - valid_frames_idx[-1]
            )

        assert all(lm is not None for lm in landmarks), "not every frame has landmark"

        return landmarks
