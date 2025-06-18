## Ohne KeyPoints

import math
import cv2
import mediapipe as mp
from ultralytics import YOLO


class PositionDetector:
    """
    Detects body position (Standing, Sitting, Lying, etc.) using YOLO v8 Pose,
    and checks eye state (open/closed) using Mediapipe FaceMesh.
    """

    def __init__(self, pose_model_path="yolov8n-pose.pt"):
        """
        Initialize the PositionDetector with a YOLO v8 pose model
        and configure a Mediapipe FaceMesh.

        Args:
            pose_model_path (str): Path to the YOLO v8 pose model file.
        """
        # YOLO Pose model
        self.pose_model = YOLO(pose_model_path)

        # Mediapipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def detect_position_and_eyes(self, frame, frame_for_display):
        """
        Analyze a frame to detect the user's posture (Standing, Sitting, Lying, etc.)
        and determine whether eyes are open or closed.

        The method uses YOLO v8 Pose to locate body keypoints and draws them
        on 'frame_for_display'. It also uses Mediapipe FaceMesh to analyze
        the face region and compute the eye aspect ratio.

        Args:
            frame (numpy.ndarray): The input frame from the camera.
            frame_for_display (numpy.ndarray): A copy of the input frame where
                                               the detected keypoints will be drawn.

        Returns:
            tuple: (position_detected, sleeping_bool)
                position_detected (str): A string describing the detected body position.
                sleeping_bool (bool): True if eyes are detected as closed, False otherwise.
        """
        results = self.pose_model(frame, save=False, task="pose")

        position_detected = "Unknown"
        eyes_status = "unbekannt"  # this value is updated after eye analysis

        for result in results:
            # Wenn gar nichts erkannt wurde, weitermachen
            if result.keypoints is None or len(result.keypoints) == 0:
                continue
            ######## test_ohne keypoints!!
            for data in result.keypoints:
                xy_array = data.xy.cpu().numpy()

                # xy_array.shape ist typischerweise (1, 17, 3), kann aber (0, x, x) sein, wenn keine Keypoints vorhanden
                if xy_array.shape[0] == 0:
                    continue  # keine Keypoints -> diesen Durchgang überspringen

                # Jetzt haben wir mindestens 1 Pose -> xy_array[0] sollte existieren
                xy_np = xy_array[0]  # shape (17,3)

                # Prüfe auch hier noch einmal, ob tatsächlich 17 Keypoints vorhanden sind:
                if xy_np.shape[0] < 17:
                    continue  # Sicherheitshalber. Oder du gehst nur weiter, wenn du wirklich 17 Keypoints erwartest
                ###########
                # Extract relevant keypoint coordinates
                nose_x, nose_y = xy_np[0][:2]
                left_shoulder_y = xy_np[5][1]
                right_shoulder_y = xy_np[6][1]
                left_hip_y = xy_np[11][1]
                right_hip_y = xy_np[12][1]
                left_knee_y = xy_np[13][1]
                right_knee_y = xy_np[14][1]

                # Define a region around the nose for face cropping
                face_size = 80
                fx1 = max(0, int(nose_x - face_size))
                fy1 = max(0, int(nose_y - face_size))
                fx2 = min(frame.shape[1], int(nose_x + face_size))
                fy2 = min(frame.shape[0], int(nose_y + face_size))

                face_crop = frame[fy1:fy2, fx1:fx2]
                face_detected = False

                # If the crop is valid, convert it to RGB and process with FaceMesh
                if face_crop.size > 0:
                    rgb_face_crop = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
                    face_results = self.face_mesh.process(rgb_face_crop)
                    face_detected = (face_results.multi_face_landmarks is not None)
                    if face_detected:
                        face_landmarks = face_results.multi_face_landmarks[0]
                        h, w, _ = face_crop.shape
                        mesh_points = []

                        # Collect all face landmarks in pixel coordinates
                        for lm in face_landmarks.landmark:
                            x_px = int(lm.x * w)
                            y_px = int(lm.y * h)
                            mesh_points.append((x_px, y_px))

                        # Indices for right and left eye landmarks in FaceMesh
                        right_eye_idx = [33, 160, 158, 133, 153, 144]
                        left_eye_idx = [362, 385, 387, 263, 373, 380]

                        # Check if indices are valid for the detected face
                        if all(i < len(mesh_points) for i in right_eye_idx + left_eye_idx):
                            ear_right = self.eye_aspect_ratio(mesh_points, right_eye_idx)
                            ear_left = self.eye_aspect_ratio(mesh_points, left_eye_idx)
                            ear_avg = (ear_right + ear_left) / 2.0
                            # Simple threshold check for open/closed eyes
                            eyes_status = "Geschlossen" if ear_avg < 0.26 else "Offen"

                # Estimate body position by comparing distances between shoulders, hips, and knees
                body_height_left = abs(left_shoulder_y - left_knee_y)
                body_height_right = abs(right_shoulder_y - right_knee_y)
                avg_body_height = (body_height_left + body_height_right) / 2.0
                if avg_body_height < 1:
                    avg_body_height = 1

                norm_shoulder_hip_left = abs(left_shoulder_y - left_hip_y) / avg_body_height
                norm_shoulder_hip_right = abs(right_shoulder_y - right_hip_y) / avg_body_height
                norm_hip_knee_left = abs(left_hip_y - left_knee_y) / avg_body_height
                norm_hip_knee_right = abs(right_hip_y - right_knee_y) / avg_body_height

                # Logic for classifying body position based on these normalized distances
                if (norm_shoulder_hip_left > 0.4 and norm_shoulder_hip_right > 0.4 and
                        norm_hip_knee_left > 0.4 and norm_hip_knee_right > 0.4):
                    position_detected = "Standing"
                elif ((norm_shoulder_hip_left < 0.2 and norm_shoulder_hip_right < 0.2) or
                      (norm_hip_knee_left < 0.2 and norm_hip_knee_right < 0.2)):
                    # Check the difference in shoulder height to guess lying side
                    delta_shoulder_y = left_shoulder_y - right_shoulder_y
                    side_threshold = 10
                    if delta_shoulder_y > side_threshold:
                        position_detected = "Lying (Left side)"
                    elif delta_shoulder_y < -side_threshold:
                        position_detected = "Lying (Right side)"
                    else:
                        # Decide between Lying (Back) vs. Lying (Belly)
                        if face_detected:
                            position_detected = "Lying (Back)"
                        else:
                            position_detected = "Lying (Belly)"

                # elif (norm_shoulder_hip_left > 0.3 and norm_shoulder_hip_right > 0.3 and #test
                     # norm_hip_knee_left < 0.3 and norm_hip_knee_right < 0.3): #test
                    # position_detected = "Sitting" #test
                else:
                    position_detected = "Sitting"

        sleeping = (eyes_status == "Geschlossen")
        return position_detected, sleeping

    @staticmethod
    def eye_aspect_ratio(landmarks, eye_indices):
        """
        Calculate the Eye Aspect Ratio (EAR) for an eye based on
        specific landmark indices.

        EAR formula:
            EAR = (dist(p1, p5) + dist(p2, p4)) / (2.0 * dist(p0, p3))

        Args:
            landmarks (list): A list of (x, y) tuples for each face landmark.
            eye_indices (list): Indices of the landmarks corresponding to a single eye.

        Returns:
            float: The computed EAR value.
        """
        (p0x, p0y) = landmarks[eye_indices[0]]
        (p1x, p1y) = landmarks[eye_indices[1]]
        (p2x, p2y) = landmarks[eye_indices[2]]
        (p3x, p3y) = landmarks[eye_indices[3]]
        (p4x, p4y) = landmarks[eye_indices[4]]
        (p5x, p5y) = landmarks[eye_indices[5]]

        # Distances between relevant points
        A = math.dist((p1x, p1y), (p5x, p5y))
        B = math.dist((p2x, p2y), (p4x, p4y))
        C = math.dist((p0x, p0y), (p3x, p3y))

        # Return the average of the vertical distances divided by the horizontal distance
        return (A + B) / (2.0 * C)
