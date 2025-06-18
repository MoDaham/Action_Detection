import math
import cv2
import mediapipe as mp
from ultralytics import YOLO
import pyrealsense2 as rs
import threading
import numpy as np
import cv2
import queue


class DrinkDetector:
    """
    Erkennt Tassen mit YOLO, berechnet den Abstand zu den Augen
    und erhöht einen Counter, wenn Mund und Tasse nah sind.
    """

    def __init__(self, cup_model_path="yolov8s.pt"):
        self.cup_model = YOLO(cup_model_path)

        self.mp_pose = mp.solutions.pose
        self.pose_detector = self.mp_pose.Pose()

        # Zählvariablen
        self.drink_count = 0
        self.is_currently_drinking = False

        # Schwellenwert in Pixeln mit Prüfung
        self.set_drinking_distance_threshold(60)

    def set_drinking_distance_threshold(self, value):
        if not isinstance(value, int):
            raise ValueError("Der Schwellenwert muss eine Ganzzahl sein.")
        if value <= 0:
            raise ValueError("Der Schwellenwert muss größer als 0 sein.")
        self.drinking_distance_threshold = value

    def check_drinking(self, color_frame, frame_for_display, depth_image):
        results = self.cup_model.predict(color_frame, conf=0.5, verbose=False)
        boxes = results[0].boxes if len(results) > 0 else []

        cup_detected = False
        cup_top = None
        cup_distance = None

        # YOLO-Auswertung
        for box in boxes:
            cls = int(box.cls[0])
            label = self.cup_model.names[cls]
            if label == "cup":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame_for_display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cup_detected = True
              
                # obere Mitte des boxes minus eines viertel
                # cup_top = ((x1 + x2) // 2, y1 + int((y2 - y1) / 4))

                # acte Mitte des boxes
                cup_top = ((x1 + x2) // 2, int((y1 + y2) // 10) + y1)

                # Mitte des boxes
                # cup_top = ((x1 + x2) // 2,  (y1 + y2) // 2 )  # schwer zu betrachten

                # obere Mitte des boxes
                # cup_top = ((x1 + x2) // 2,  y1)
                # das bringt meistens zu fehlern beim lesen
                break

        if cup_detected:
            rgb_frame = cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
            pose_results = self.pose_detector.process(rgb_frame)
            if pose_results.pose_landmarks:
                lms = pose_results.pose_landmarks.landmark
                h, w, _ = color_frame.shape

                # --- AUGENPOSITION STATT MUND ---
                left_eye = lms[self.mp_pose.PoseLandmark.LEFT_EYE]
                right_eye = lms[self.mp_pose.PoseLandmark.RIGHT_EYE]

                # Umrechung in Pixelkoordinaten
                lex, ley = int(left_eye.x * w), int(left_eye.y * h)
                rex, rey = int(right_eye.x * w), int(right_eye.y * h)

                # Mittelpunkt zwischen linkem und rechtem Auge
                eye_x = (lex + rex) // 2
                eye_y = (ley + rey) // 2
                eye_center = (eye_x, eye_y)

                # Prüfe erst, ob die Koordinaten im gültigen Bereich liegen
                if 0 <= eye_center[1] < depth_image.shape[0] and 0 <= eye_center[0] < depth_image.shape[1]:
                    eye_depth_value = depth_image[eye_center[1], eye_center[0]]
                else:
                    eye_depth_value = 0  

                if 0 <= cup_top[1] < depth_image.shape[0] and 0 <= cup_top[0] < depth_image.shape[1]:
                    cup_depth_value = depth_image[cup_top[1], cup_top[0]]
                else:
                    cup_depth_value = 0

                # Zur Visualisierung
                cv2.circle(frame_for_display, (lex, ley), 5, (255, 0, 0), -1)
                cv2.circle(frame_for_display, (rex, rey), 5, (255, 0, 0), -1)
                cv2.circle(frame_for_display, eye_center, 5, (0, 0, 255), -1)

                # Abstand berechnen zwischen Augen-Mittelpunkt und oberer Kante des Cups
                dist = int(self._calculate_distance(eye_center, cup_top))
                dist_3d = self._calculate_3d_distance(eye_center, eye_depth_value, cup_top, cup_depth_value)

                # Prüfen, ob unterhalb des Schwellenwertes
            if eye_depth_value != 0 and cup_depth_value != 0:
                # Umrechnung in Meter
                eye_distance = eye_depth_value * 0.001
                cup_distance = cup_depth_value * 0.001
                dist_3d = dist_3d * 0.001
                print(
                    f"Tasse: {cup_distance:.3f} m ,       die Augen:{eye_distance:.3f}m ,    distance:{dist},       distance in m {dist_3d:.3f}")
                if (dist_3d) < 0.15 and dist_3d != 0:
                    if not self.is_currently_drinking:
                        self.drink_count += 1
                        self.is_currently_drinking = True
                        print(
                            f"Trinken erkannt! Entfernung zur Tasse: {cup_distance:.3f} m , zu den Augen:{eye_distance:.3f}")
                else:
                    self.is_currently_drinking = False  # Sicherstellen, dass kein Trinken erkannt wird
            else:
                print("Warnung: Ungültige Depth-Werte – Erkennung übersprungen")
                self.is_currently_drinking = False
        else:
            self.is_currently_drinking = False

        return self.drink_count

    def reset(self):
        """Setzt den Zähler zurück, z.B. beim Kamera-Neustart."""
        self.drink_count = 0
        self.is_currently_drinking = False

    def get_drink_count(self):
        """Gibt den aktuellen Zählerstand zurück."""
        return self.drink_count

    @staticmethod
    def _calculate_distance(p1, p2):
        if p1 is None or p2 is None:
            return 999999
        return math.dist(p1, p2)

    def _calculate_3d_distance(self, eye_center, eye_distance, cup_top, cup_distance):
        # Berechne die räumliche Distanz unter Einbeziehung der Tiefeninformationen.
        eye_x, eye_y = eye_center
        cup_x, cup_y = cup_top
        dx = eye_x - cup_x
        dy = eye_y - cup_y
        dz = eye_distance - cup_distance
        # Stelle sicher, dass die Distanz nicht negativ wird
        if (eye_y < cup_y):
            distance_squared = max(0, dx ** 2 + dy ** 2 + dz ** 2)
            return math.sqrt(distance_squared)
        else:
            print("cup ist über die auge")
            self.is_currently_drinking = False
            return self.is_currently_drinking
