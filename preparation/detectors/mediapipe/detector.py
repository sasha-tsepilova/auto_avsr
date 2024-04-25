#! /usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2021 Imperial College London (Pingchuan Ma)
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import mediapipe as mp
import numpy as np
import cv2

from detectors.mediapipe.face_geometry import PCF, get_metric_landmarks


class LandmarksDetector:
    def __init__(self):
        self.landmark_detector =  mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.lip_indeces = [0, 13, 14, 17, 37, 39, 40, 61, 78, 80, 81, 82, 84, 87, 88, 91, 95, 146, 178, 181, 185, 191, 267, 269, 270, 291, 308, 310, 311, 312, 314, 317, 318, 321, 324, 375, 402, 405, 409, 415]

    def __call__(self, video_frames):
        
        landmarks = self.detect(video_frames)
        assert any(l is not None for l in landmarks), "Cannot detect any frames in the video"
        return landmarks

    def detect(self, video_frames):
        landmarks = []
        for frame in video_frames:
            frame_height, frame_width, channels = frame.shape

            focal_length = frame_width
            center = (frame_width / 2, frame_height / 2)
            camera_matrix = np.array(
                [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
                dtype="double",
            )

            dist_coeff = np.zeros((4, 1))

            pcf = PCF(
                near=1,
                far=10000,
                frame_height=frame_height,
                frame_width=frame_width,
                fy=camera_matrix[1, 1],
            )
            results = self.landmark_detector.process(frame)
            if not results.multi_face_landmarks:
                landmarks.append(None)
                continue
            face_landmarks = results.multi_face_landmarks[0]
            face_points = np.array(
                [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
            )

            face_points = face_points.T
            face_points = face_points[:, :468]

            metric_landmarks, pose_transform_mat = get_metric_landmarks(
                face_points.copy(), pcf
            )
            model_points = metric_landmarks[0:3, self.lip_indeces].T
            pose_transform_mat[1:3, :] = -pose_transform_mat[1:3, :]

            identity_rotation = np.identity(3)
            identity_rotation[2][2] = -1
            identity_rotation[1][1] = -1
            identity_translation = np.zeros((3,1))
            identity_translation[2] = 50
            projected_model_points, jacobian_projected = cv2.projectPoints(
                    model_points,
                    identity_rotation,
                    identity_translation,
                    camera_matrix,
                    dist_coeff,
            )
            projected_model_points = [projection[0] for projection in projected_model_points]
            landmarks.append(projected_model_points)
        return landmarks
