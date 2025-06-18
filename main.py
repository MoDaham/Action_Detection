import tkinter as tk
from tkinter import Label, Frame, Menu
import threading
import queue
import numpy as np
import cv2
import pyrealsense2 as rs
from PIL import Image, ImageTk
import time  # Time library for sleep timer

# Custom classes
from modules.language import Language
from modules.drink_detector import DrinkDetector
from modules.Pose_test import PositionDetector  # Mit keypoints
# from modules.position_detector import PositionDetector # Ohne keypoints, um Absturtzen des Programms zu vermeiden


class PositionApp:
    """
    Main application class for displaying the camera feed, detecting body position,
    determining sleep/awake state, and counting drinks.
    """

    def __init__(self, root):
        """
        Initialize the PositionApp with the given Tkinter root window.
        Sets up the GUI components, initializes detectors, and configures the default language.
        """
        self.root = root
        # Default language set to German
        self.language = Language("DE")

        # Window configuration
        self.root.title(self.language.t("app_title"))
        self.root.geometry("1100x600")
        self.root.configure(bg="#1e1e1e")

        # Load an icon image from "images\\icon_htw.png" and set it as window icon
        icon_image = Image.open("images\\icon_htw.png").resize((32, 32))
        icon_photo = ImageTk.PhotoImage(icon_image)
        self.root.iconphoto(False, icon_photo)

        # Create and attach the menu
        self.build_menu()

        # Camera frame
        camera_frame = Frame(root, bg="#333", relief="ridge", bd=5)
        camera_frame.place(relx=0.05, rely=0.1, relwidth=0.7, relheight=0.7)
        self.camera_label = Label(camera_frame, bg="black")
        self.camera_label.pack(expand=True, fill="both")

        # Status frame (LED indicators)
        status_frame = Frame(root, bg="#1e1e1e")
        status_frame.place(relx=0.8, rely=0.1, relwidth=0.30, relheight=0.9)

        # Possible statuses
        self.statuses = [
            "Standing",
            "Sitting",
            "Lying (Left side)",
            "Lying (Right side)",
            "Lying (Back)",
            "Lying (Belly)",
            "Sleeping",
            "Awake"
        ]
        # Dictionary to hold references to LED labels
        self.led_labels = {}
        for i, status in enumerate(self.statuses):
            label = Label(
                status_frame,
                text=self.language.t(status),
                font=("Arial", 14, "bold"),
                bg="#555",
                fg="white",
                width=15,
                height=2,
                wraplength=150
            )
            label.grid(row=i, column=0, pady=5, padx=5)
            self.led_labels[status] = label

        # Label for drink counter
        self.drink_label = Label(
            status_frame,
            text=f"{self.language.t('drink_label')}0",
            font=("Arial", 14, "bold"),
            bg="#555",
            fg="white",
            width=15,
            height=2
        )
        self.drink_label.grid(row=len(self.statuses), column=0, pady=5, padx=5)

        # Button frame
        button_frame = Frame(root, bg="#1e1e1e")
        button_frame.place(relx=0.05, rely=0.85, relwidth=0.7, relheight=0.1)

        # Start (Position) button
        self.start_button = tk.Button(
            button_frame,
            text=self.language.t("start"),
            command=self.start_position_camera,  # Limited to position detection
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12, "bold")
        )
        self.start_button.pack(side="left", padx=20, pady=10)

        # Stop button
        self.stop_button = tk.Button(
            button_frame,
            text=self.language.t("stop"),
            command=self.stop_camera,
            bg="#E74C3C",
            fg="white",
            font=("Arial", 12, "bold")
        )
        self.stop_button.pack(side="left", padx=20, pady=10)

        # Drinking mode button
        self.drinking_button = tk.Button(
            button_frame,
            text=self.language.t("drinking"),
            command=self.start_drinking_camera,  # Limited to drinking behavior
            bg="#1E90FF",
            fg="white",
            font=("Arial", 12, "bold")
        )
        self.drinking_button.pack(side="left", padx=20, pady=10)

        # Detector classes
        self.position_detector = PositionDetector("yolov8n-pose.pt")
        self.drink_detector = DrinkDetector("yolov8s.pt")

        # Camera capture object
        self.cap = None
        self.running = False
        self.pipe = None  # RealSense pipeline object
        self.mode = None  # Distinguish which mode is active: "position" or "drinking"

        # Queues for communication between worker threads and GUI
        self.frame_queue_position = queue.Queue()
        self.frame_queue_drinking = queue.Queue()

        # Sleep timer variable to confirm eyes have been closed for 10 seconds
        self.sleep_timer_start = None

    # ----------------------------
    #   MENU AND LANGUAGES
    # ----------------------------
    def build_menu(self):
        """
        Build the main menu, including the settings and language submenus,
        and attach it to the main window.
        """
        self.menubar = Menu(self.root)
        self.settings_menu = Menu(self.menubar, tearoff=0)
        self.language_menu = Menu(self.settings_menu, tearoff=0)

        # Language selection
        self.language_menu.add_command(
            label=self.language.t("menu_german"),
            command=lambda: self.change_language("DE")
        )
        self.language_menu.add_command(
            label=self.language.t("menu_english"),
            command=lambda: self.change_language("EN")
        )

        self.settings_menu.add_cascade(
            label=self.language.t("menu_language"),
            menu=self.language_menu
        )
        self.menubar.add_cascade(
            label=self.language.t("menu_settings"),
            menu=self.settings_menu
        )

        self.root.config(menu=self.menubar)

    def change_language(self, lang):
        """
        Change the current language of the application and update all GUI texts.
        """
        self.language.set_language(lang)
        self.update_texts()

    def update_texts(self):
        """
        Refresh all on-screen texts (window title, buttons, menu, LED labels, etc.)
        according to the current language.
        """
        # Window title
        self.root.title(self.language.t("app_title"))

        # Rebuild the entire menu so the entries are updated with the new language
        self.build_menu()

        # Update buttons
        self.start_button.config(text=self.language.t("start"))
        self.stop_button.config(text=self.language.t("stop"))
        self.drinking_button.config(text=self.language.t("drinking"))

        # Update LED labels
        for status in self.statuses:
            self.led_labels[status].config(text=self.language.t(status))

        # Update the drink label (current count)
        current_count = self.drink_detector.get_drink_count()
        self.update_drink_label(current_count)

    # ----------------------------
    #   START / STOP
    # ----------------------------
    def start_position_camera(self):
        """
        Start the camera in position detection mode in a separate thread.
        The GUI is updated via .after(...).
        """
        if self.running:
            print("Camera is already running.")
            return

        self.running = True
        self.mode = "position"
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Could not open the camera.")
            self.running = False
            return

        self.reset_ui()

        # Worker thread for camera + position detection
        threading.Thread(target=self.camera_loop_position, daemon=True).start()

        # Start the GUI update for position mode
        self.update_gui_position()
        print("Camera started for POSITION detection.")

    def start_drinking_camera(self):
        """
        Start the camera in drinking detection mode in a separate thread.
        The GUI is updated via .after(...).
        """
        if self.running:
            print("Camera is already running.")
            return

        self.running = True
        self.mode = "drinking"

        # Create RealSense Pipeline
        self.pipe = rs.pipeline()
        cfg = rs.config()

        # Enable depth and color streams
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

        # Start the pipeline
        self.pipe.start(cfg)

        self.reset_ui()

        # Worker thread for drinking detection
        threading.Thread(target=self.camera_loop_drinking, daemon=True).start()

        # Start the GUI update for drinking mode
        self.update_gui_drinking()
        print("Camera started for DRINKING detection (RealSense).")

    def stop_camera(self):
        """
        Stop the camera capture, release resources, and reset the UI.
        """
        self.running = False
        self.mode = None

        if self.cap and self.cap.isOpened():
            self.cap.release()

        if self.pipe:
            try:
                self.pipe.stop()
            except:
                pass
            self.pipe = None

        self.camera_label.config(image="")
        print("Camera stopped.")
        self.reset_ui()

    # ----------------------------
    #   WORKER THREADS
    # ----------------------------
    def camera_loop_position(self):
        """
        Runs in a separate thread.
        Reads frames from the camera, performs position & sleep detection,
        and stores the processed frame in self.frame_queue_position.
        """
        while self.running and self.cap.isOpened() and self.mode == "position":
            ret, frame = self.cap.read()
            if not ret:
                break

            frame_for_display = frame.copy()

            # Position & sleep detection
            position, sleeping = self.position_detector.detect_position_and_eyes(
                frame, frame_for_display
            )

            # -------------------------------
            # 10-second timer logic for "Sleeping" confirmation
            # -------------------------------
            if sleeping:
                if self.sleep_timer_start is None:
                    self.sleep_timer_start = time.time()
                else:
                    elapsed = time.time() - self.sleep_timer_start
                    # If eyes have been closed for >= 10 seconds, confirm "Sleeping"
                    if elapsed >= 10:
                        sleeping = True
                    else:
                        sleeping = False
            else:
                # Eyes are open -> reset timer
                self.sleep_timer_start = None

            # Put the processed frame along with position/sleeping info into the queue
            self.frame_queue_position.put((frame_for_display, position, sleeping))

        if self.cap and self.cap.isOpened():
            self.cap.release()
        print("camera_loop_position has ended.")

    def camera_loop_drinking(self):
        """
        Runs in a separate thread.
        Reads frames from the RealSense camera, performs drinking detection,
        and stores the processed frame in self.frame_queue_drinking.
        """
        while self.running and self.mode == "drinking":
            # Get frames from RealSense
            frames = self.pipe.wait_for_frames()
            if not frames:
                break

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()

            if not depth_frame or not color_frame:
                print("No depth values found.")
                continue

            # Align depth to the color stream
            frames = rs.align(rs.stream.color).process(frames)

            # Convert frames to NumPy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            # Use the color image for display
            frame_for_display = color_image.copy()

            # Drinking detection
            count = self.drink_detector.check_drinking(color_image, frame_for_display, depth_image)

            # Put the processed frame and count into the queue
            self.frame_queue_drinking.put((frame_for_display, count))

        if self.pipe:
            self.pipe.stop()
        print("camera_loop_drinking has ended.")

    # ----------------------------
    #   GUI UPDATE (via .after)
    # ----------------------------
    def update_gui_position(self):
        """
        Fetches the latest data from frame_queue_position and updates
        the Tkinter label and LED indicators (Position/Sleep).
        Called periodically via .after(...) as long as self.running=True and self.mode="position".
        """
        if not self.running or self.mode != "position":
            return

        try:
            # If there is something in the queue, take the newest element
            frame_for_display, position, sleeping = self.frame_queue_position.get_nowait()
        except queue.Empty:
            # No new frames
            frame_for_display, position, sleeping = None, None, None

        # Update the GUI only if we have a new image
        if frame_for_display is not None:
            # Update LED indicators
            self.update_position_leds(position, sleeping)

            # Show image in Tkinter
            rgb_frame = cv2.cvtColor(frame_for_display, cv2.COLOR_BGR2RGB)
            img_tk = ImageTk.PhotoImage(image=Image.fromarray(rgb_frame))
            self.camera_label.imgtk = img_tk
            self.camera_label.config(image=img_tk)

        # Try again in 10 ms
        self.root.after(10, self.update_gui_position)

    def update_gui_drinking(self):
        """
        Fetches the latest data from frame_queue_drinking and updates
        the Tkinter label and the drink count.
        Called periodically via .after(...) as long as self.running=True and self.mode="drinking".
        """
        if not self.running or self.mode != "drinking":
            return

        try:
            frame_for_display, count = self.frame_queue_drinking.get_nowait()
        except queue.Empty:
            frame_for_display, count = None, None

        if frame_for_display is not None:
            # Update drink label
            if count is not None:
                self.update_drink_label(count)

            # Show image in Tkinter
            rgb_frame = cv2.cvtColor(frame_for_display, cv2.COLOR_BGR2RGB)
            img_tk = ImageTk.PhotoImage(image=Image.fromarray(rgb_frame))
            self.camera_label.imgtk = img_tk
            self.camera_label.config(image=img_tk)

        # Try again in 10 ms
        self.root.after(10, self.update_gui_drinking)

    # ----------------------------
    #   UI UPDATES
    # ----------------------------
    def reset_ui(self):
        """
        Reset all LED states and drink counter to their default values.
        Also reset the sleep timer.
        """
        # Reset all LEDs
        for status in self.statuses:
            self.led_labels[status].config(bg="#555")

        # Reset the drink count
        self.drink_detector.reset()
        self.update_drink_label(0)

        # Reset sleep timer
        self.sleep_timer_start = None

    def update_position_leds(self, position_detected, is_sleeping):
        """
        Update the LED labels based on the detected position and sleeping state.
        """
        if position_detected is None or is_sleeping is None:
            return

        # Reset all to default
        for status in self.statuses:
            self.led_labels[status].config(bg="#555")

        # Update sleeping/awake LEDs
        if is_sleeping:
            self.led_labels["Sleeping"].config(bg="red")
        else:
            self.led_labels["Awake"].config(bg="green")

        # Show the specific position in green
        if position_detected in self.led_labels:
            self.led_labels[position_detected].config(bg="green")

    def update_drink_label(self, count):
        """
        Update the drink label text and color based on the current drink count.
        """
        label_text = f"{self.language.t('drink_label')}{count}"
        if count > 0:
            self.drink_label.config(text=label_text, bg="blue")
        else:
            self.drink_label.config(text=label_text, bg="#555")


# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = PositionApp(root)
    root.mainloop()
