"""
Door Entry System Kiosk
=======================
A full-screen face recognition door entry system with:
- Real-time face recognition
- Access granted/denied visual feedback
- Access log with timestamps
- Admin panel for face registration and training
- Door control simulation (with GPIO hooks for Raspberry Pi)

Compatible with standard webcams (OpenCV) and Raspberry Pi Camera (picamera2)
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import cv2
import os
import threading
import time
from datetime import datetime
import pickle
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFont
import json

# Try to import face_recognition - required for this application
try:
    import face_recognition
except ImportError:
    print("Error: face_recognition library not found. Please install it with: pip install face-recognition")
    exit(1)

# Try to import picamera2 for Raspberry Pi, fall back to OpenCV
USE_PICAMERA = False
try:
    from picamera2 import Picamera2
    USE_PICAMERA = True
except ImportError:
    USE_PICAMERA = False

# Try to import GPIO for Raspberry Pi door control
USE_GPIO = False
try:
    import RPi.GPIO as GPIO
    USE_GPIO = True
except ImportError:
    USE_GPIO = False


# ==================== CONFIGURATION ====================
class Config:
    # Kiosk Settings
    FULLSCREEN = True
    WINDOW_TITLE = "Door Entry System"
    
    # Admin Settings
    ADMIN_PASSWORD = "admin123"  # Change this in production!
    
    # Camera Settings
    CAMERA_RESOLUTION = (640, 480)
    
    # Recognition Settings
    RECOGNITION_THRESHOLD = 0.5  # Lower = more strict (0.0 - 1.0)
    COOLDOWN_SECONDS = 5  # Prevent repeated access logs for same person
    
    # Door Control (GPIO Pin for Raspberry Pi)
    DOOR_RELAY_PIN = 17
    DOOR_UNLOCK_DURATION = 5  # Seconds to keep door unlocked
    
    # File Paths
    DATASET_PATH = "dataset"
    ENCODINGS_PATH = "encodings.pickle"
    ACCESS_LOG_PATH = "access_log.json"
    
    # Colors (RGB)
    COLOR_GRANTED = "#00C853"  # Green
    COLOR_DENIED = "#FF1744"   # Red
    COLOR_SCANNING = "#2196F3" # Blue
    COLOR_DARK_BG = "#1a1a2e"
    COLOR_PANEL_BG = "#16213e"
    COLOR_TEXT = "#ffffff"
    COLOR_TEXT_DIM = "#888888"


# ==================== DOOR CONTROLLER ====================
class DoorController:
    """Controls the physical door lock (GPIO for Raspberry Pi or simulation)"""
    
    def __init__(self):
        self.is_unlocked = False
        self.unlock_thread = None
        
        if USE_GPIO:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(Config.DOOR_RELAY_PIN, GPIO.OUT)
            GPIO.output(Config.DOOR_RELAY_PIN, GPIO.LOW)
    
    def unlock(self, duration=None):
        """Unlock the door for specified duration"""
        if duration is None:
            duration = Config.DOOR_UNLOCK_DURATION
        
        if self.unlock_thread and self.unlock_thread.is_alive():
            return  # Already unlocking
        
        self.unlock_thread = threading.Thread(target=self._unlock_sequence, args=(duration,))
        self.unlock_thread.start()
    
    def _unlock_sequence(self, duration):
        """Execute the unlock sequence"""
        self.is_unlocked = True
        
        if USE_GPIO:
            GPIO.output(Config.DOOR_RELAY_PIN, GPIO.HIGH)
        
        print(f"[DOOR] Unlocked for {duration} seconds")
        time.sleep(duration)
        
        if USE_GPIO:
            GPIO.output(Config.DOOR_RELAY_PIN, GPIO.LOW)
        
        self.is_unlocked = False
        print("[DOOR] Locked")
    
    def cleanup(self):
        """Cleanup GPIO on exit"""
        if USE_GPIO:
            GPIO.cleanup()


# ==================== ACCESS LOG ====================
class AccessLog:
    """Manages access log entries"""
    
    def __init__(self, log_path=None):
        self.log_path = log_path or Config.ACCESS_LOG_PATH
        self.entries = []
        self.load()
    
    def load(self):
        """Load existing log entries"""
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, 'r') as f:
                    self.entries = json.load(f)
            except:
                self.entries = []
    
    def save(self):
        """Save log entries to file"""
        with open(self.log_path, 'w') as f:
            json.dump(self.entries, f, indent=2)
    
    def add_entry(self, name, access_granted, confidence=0.0):
        """Add a new access log entry"""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "name": name,
            "access_granted": access_granted,
            "confidence": round(confidence, 3)
        }
        self.entries.append(entry)
        self.save()
        return entry
    
    def get_recent(self, count=50):
        """Get most recent entries"""
        return list(reversed(self.entries[-count:]))
    
    def clear(self):
        """Clear all entries"""
        self.entries = []
        self.save()


# ==================== CAMERA MANAGER ====================
class CameraManager:
    """Manages camera operations for both standard webcams and Raspberry Pi camera"""
    
    def __init__(self, use_picamera=False, resolution=(640, 480)):
        self.use_picamera = use_picamera
        self.resolution = resolution
        self.camera = None
        self.is_running = False
    
    def start(self):
        """Initialize and start the camera"""
        if self.use_picamera:
            self.camera = Picamera2()
            self.camera.configure(self.camera.create_preview_configuration(
                main={"format": 'XRGB8888', "size": self.resolution}
            ))
            self.camera.start()
        else:
            self.camera = cv2.VideoCapture(0)
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
        
        self.is_running = True
        time.sleep(0.5)
    
    def capture_frame(self):
        """Capture a single frame from the camera"""
        if not self.is_running:
            return None
        
        if self.use_picamera:
            frame = self.camera.capture_array()
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        else:
            ret, frame = self.camera.read()
            if not ret:
                return None
        
        return frame
    
    def stop(self):
        """Stop and release the camera"""
        self.is_running = False
        if self.camera is not None:
            if self.use_picamera:
                self.camera.stop()
            else:
                self.camera.release()
            self.camera = None


# ==================== FACE RECOGNITION SYSTEM ====================
class FaceRecognitionSystem:
    """Core face recognition logic"""
    
    def __init__(self, dataset_path=None, encodings_path=None):
        self.dataset_path = dataset_path or Config.DATASET_PATH
        self.encodings_path = encodings_path or Config.ENCODINGS_PATH
        self.known_encodings = []
        self.known_names = []
        self.cv_scaler = 4
        
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
        
        self.load_encodings()
    
    def load_encodings(self):
        """Load face encodings from pickle file"""
        if os.path.exists(self.encodings_path):
            try:
                with open(self.encodings_path, "rb") as f:
                    data = pickle.loads(f.read())
                self.known_encodings = data["encodings"]
                self.known_names = data["names"]
                return True
            except Exception as e:
                print(f"Error loading encodings: {e}")
                return False
        return False
    
    def get_registered_persons(self):
        """Get list of registered persons from dataset folder"""
        persons = []
        if os.path.exists(self.dataset_path):
            for name in os.listdir(self.dataset_path):
                person_path = os.path.join(self.dataset_path, name)
                if os.path.isdir(person_path):
                    image_count = len([f for f in os.listdir(person_path) 
                                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
                    persons.append((name, image_count))
        return persons
    
    def get_trained_persons(self):
        """Get unique names from trained encodings"""
        return list(set(self.known_names))
    
    def create_person_folder(self, name):
        """Create a folder for a new person"""
        person_folder = os.path.join(self.dataset_path, name)
        if not os.path.exists(person_folder):
            os.makedirs(person_folder)
        return person_folder
    
    def save_face_image(self, frame, name):
        """Save a captured face image"""
        folder = self.create_person_folder(name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{name}_{timestamp}.jpg"
        filepath = os.path.join(folder, filename)
        cv2.imwrite(filepath, frame)
        return filepath
    
    def train_model(self, progress_callback=None):
        """Train the face recognition model on all images in dataset"""
        from imutils import paths
        
        image_paths = list(paths.list_images(self.dataset_path))
        if not image_paths:
            return False, "No images found in dataset folder"
        
        known_encodings = []
        known_names = []
        
        for i, image_path in enumerate(image_paths):
            if progress_callback:
                progress_callback(i + 1, len(image_paths), image_path)
            
            name = image_path.split(os.path.sep)[-2]
            
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            boxes = face_recognition.face_locations(rgb, model="hog")
            encodings = face_recognition.face_encodings(rgb, boxes)
            
            for encoding in encodings:
                known_encodings.append(encoding)
                known_names.append(name)
        
        if not known_encodings:
            return False, "No faces detected in any images"
        
        data = {"encodings": known_encodings, "names": known_names}
        with open(self.encodings_path, "wb") as f:
            f.write(pickle.dumps(data))
        
        self.known_encodings = known_encodings
        self.known_names = known_names
        
        return True, f"Training complete! {len(known_encodings)} encodings from {len(set(known_names))} persons"
    
    def recognize_faces(self, frame):
        """Detect and recognize faces in a frame"""
        if not self.known_encodings:
            return frame, []
        
        small_frame = cv2.resize(frame, (0, 0), fx=1/self.cv_scaler, fy=1/self.cv_scaler)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
        
        results = []
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            top *= self.cv_scaler
            right *= self.cv_scaler
            bottom *= self.cv_scaler
            left *= self.cv_scaler
            
            matches = face_recognition.compare_faces(self.known_encodings, face_encoding)
            name = "Unknown"
            confidence = 0.0
            
            if True in matches:
                face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = self.known_names[best_match_index]
                    confidence = 1 - face_distances[best_match_index]
            
            results.append({
                "name": name,
                "confidence": confidence,
                "location": (top, right, bottom, left)
            })
        
        return frame, results


# ==================== KIOSK GUI ====================
class DoorEntryKiosk:
    """Main Kiosk Application"""
    
    def __init__(self, root):
        self.root = root
        self.root.title(Config.WINDOW_TITLE)
        
        # Set fullscreen or window size
        if Config.FULLSCREEN:
            self.root.attributes('-fullscreen', True)
            self.root.bind('<Escape>', lambda e: self.toggle_fullscreen())
        else:
            self.root.geometry("1280x800")
        
        self.root.configure(bg=Config.COLOR_DARK_BG)
        
        # Initialize components
        self.camera = CameraManager(use_picamera=USE_PICAMERA, resolution=Config.CAMERA_RESOLUTION)
        self.face_system = FaceRecognitionSystem()
        self.door_controller = DoorController()
        self.access_log = AccessLog()
        
        # State variables
        self.is_running = True
        self.is_scanning = True
        self.camera_thread = None
        self.current_status = "scanning"  # scanning, granted, denied
        self.status_message = ""
        self.last_access = {}  # Track cooldowns per person
        self.admin_mode = False
        
        # Registration state
        self.registration_mode = False
        self.registration_name = ""
        self.captured_count = 0
        self.current_frame = None
        
        # Create GUI
        self.create_kiosk_interface()
        
        # Start camera
        self.start_camera()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Bind keyboard shortcuts
        self.root.bind('<F1>', lambda e: self.show_admin_login())
        self.root.bind('<F11>', lambda e: self.toggle_fullscreen())
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        is_fullscreen = self.root.attributes('-fullscreen')
        self.root.attributes('-fullscreen', not is_fullscreen)
    
    def create_kiosk_interface(self):
        """Create the main kiosk interface"""
        # Main container
        self.main_frame = tk.Frame(self.root, bg=Config.COLOR_DARK_BG)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Header
        header_frame = tk.Frame(self.main_frame, bg=Config.COLOR_DARK_BG)
        header_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Title
        self.title_label = tk.Label(
            header_frame, 
            text="🚪 DOOR ENTRY SYSTEM", 
            font=("Helvetica", 32, "bold"),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_DARK_BG
        )
        self.title_label.pack(side=tk.LEFT)
        
        # Time display
        self.time_label = tk.Label(
            header_frame,
            text="",
            font=("Helvetica", 24),
            fg=Config.COLOR_TEXT_DIM,
            bg=Config.COLOR_DARK_BG
        )
        self.time_label.pack(side=tk.RIGHT)
        self.update_time()
        
        # Content area (camera + status)
        content_frame = tk.Frame(self.main_frame, bg=Config.COLOR_DARK_BG)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left side - Camera feed
        camera_container = tk.Frame(content_frame, bg=Config.COLOR_PANEL_BG, bd=3, relief=tk.RAISED)
        camera_container.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))
        
        self.video_label = tk.Label(camera_container, bg=Config.COLOR_PANEL_BG)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Right side - Status panel
        status_container = tk.Frame(content_frame, bg=Config.COLOR_PANEL_BG, width=400)
        status_container.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))
        status_container.pack_propagate(False)
        
        # Status indicator (large icon area)
        self.status_frame = tk.Frame(status_container, bg=Config.COLOR_PANEL_BG, height=300)
        self.status_frame.pack(fill=tk.X, padx=20, pady=20)
        self.status_frame.pack_propagate(False)
        
        self.status_icon_label = tk.Label(
            self.status_frame,
            text="👁",
            font=("Helvetica", 80),
            fg=Config.COLOR_SCANNING,
            bg=Config.COLOR_PANEL_BG
        )
        self.status_icon_label.pack(expand=True)
        
        self.status_text_label = tk.Label(
            self.status_frame,
            text="SCANNING...",
            font=("Helvetica", 24, "bold"),
            fg=Config.COLOR_SCANNING,
            bg=Config.COLOR_PANEL_BG
        )
        self.status_text_label.pack()
        
        self.status_detail_label = tk.Label(
            self.status_frame,
            text="Please look at the camera",
            font=("Helvetica", 14),
            fg=Config.COLOR_TEXT_DIM,
            bg=Config.COLOR_PANEL_BG
        )
        self.status_detail_label.pack(pady=(10, 0))
        
        # Recent access log
        log_label = tk.Label(
            status_container,
            text="Recent Access",
            font=("Helvetica", 16, "bold"),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_PANEL_BG
        )
        log_label.pack(pady=(20, 10))
        
        log_frame = tk.Frame(status_container, bg=Config.COLOR_DARK_BG)
        log_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        self.log_listbox = tk.Listbox(
            log_frame,
            font=("Consolas", 11),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_DARK_BG,
            selectbackground=Config.COLOR_PANEL_BG,
            highlightthickness=0,
            bd=0
        )
        self.log_listbox.pack(fill=tk.BOTH, expand=True)
        
        # Footer
        footer_frame = tk.Frame(self.main_frame, bg=Config.COLOR_DARK_BG)
        footer_frame.pack(fill=tk.X, pady=(20, 0))
        
        # Admin button (subtle)
        self.admin_btn = tk.Button(
            footer_frame,
            text="⚙",
            font=("Helvetica", 16),
            fg=Config.COLOR_TEXT_DIM,
            bg=Config.COLOR_DARK_BG,
            activebackground=Config.COLOR_PANEL_BG,
            activeforeground=Config.COLOR_TEXT,
            bd=0,
            command=self.show_admin_login
        )
        self.admin_btn.pack(side=tk.LEFT)
        
        # Status info
        self.info_label = tk.Label(
            footer_frame,
            text=f"Model: {len(self.face_system.get_trained_persons())} persons registered | Press F1 for admin",
            font=("Helvetica", 10),
            fg=Config.COLOR_TEXT_DIM,
            bg=Config.COLOR_DARK_BG
        )
        self.info_label.pack(side=tk.RIGHT)
        
        # Load recent log entries
        self.update_log_display()
    
    def update_time(self):
        """Update the time display"""
        current_time = datetime.now().strftime("%H:%M:%S")
        current_date = datetime.now().strftime("%A, %B %d, %Y")
        self.time_label.config(text=f"{current_date}\n{current_time}")
        self.root.after(1000, self.update_time)
    
    def update_log_display(self):
        """Update the access log display"""
        self.log_listbox.delete(0, tk.END)
        entries = self.access_log.get_recent(10)
        
        for entry in entries:
            timestamp = datetime.fromisoformat(entry['timestamp']).strftime("%H:%M:%S")
            status = "✓" if entry['access_granted'] else "✗"
            color_tag = "granted" if entry['access_granted'] else "denied"
            self.log_listbox.insert(tk.END, f"{timestamp} {status} {entry['name']}")
    
    def set_status(self, status, name="", confidence=0.0):
        """Update the status display"""
        self.current_status = status
        
        if status == "granted":
            self.status_icon_label.config(text="✓", fg=Config.COLOR_GRANTED)
            self.status_text_label.config(text="ACCESS GRANTED", fg=Config.COLOR_GRANTED)
            self.status_detail_label.config(text=f"Welcome, {name}!\nConfidence: {confidence:.1%}")
            self.status_frame.config(bg=Config.COLOR_GRANTED)
            self.root.after(3000, lambda: self.set_status("scanning"))
            
        elif status == "denied":
            self.status_icon_label.config(text="✗", fg=Config.COLOR_DENIED)
            self.status_text_label.config(text="ACCESS DENIED", fg=Config.COLOR_DENIED)
            self.status_detail_label.config(text="Unrecognized person\nContact administrator")
            self.status_frame.config(bg=Config.COLOR_DENIED)
            self.root.after(3000, lambda: self.set_status("scanning"))
            
        else:  # scanning
            self.status_icon_label.config(text="👁", fg=Config.COLOR_SCANNING)
            self.status_text_label.config(text="SCANNING...", fg=Config.COLOR_SCANNING)
            self.status_detail_label.config(text="Please look at the camera")
            self.status_frame.config(bg=Config.COLOR_PANEL_BG)
        
        # Update background colors for all children
        for widget in self.status_frame.winfo_children():
            if status in ["granted", "denied"]:
                widget.config(bg=self.status_frame.cget('bg'))
            else:
                widget.config(bg=Config.COLOR_PANEL_BG)
    
    def start_camera(self):
        """Start the camera in a background thread"""
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()
    
    def camera_loop(self):
        """Main camera loop running in a separate thread"""
        try:
            self.camera.start()
            
            while self.is_running:
                frame = self.camera.capture_frame()
                if frame is None:
                    continue
                
                display_frame = frame.copy()
                
                if self.is_scanning and not self.registration_mode:
                    # Perform face recognition
                    _, results = self.face_system.recognize_faces(frame)
                    
                    for result in results:
                        top, right, bottom, left = result['location']
                        name = result['name']
                        confidence = result['confidence']
                        
                        # Determine access
                        if name != "Unknown" and confidence >= Config.RECOGNITION_THRESHOLD:
                            # Check cooldown
                            now = time.time()
                            if name not in self.last_access or (now - self.last_access[name]) > Config.COOLDOWN_SECONDS:
                                self.last_access[name] = now
                                
                                # Grant access
                                self.root.after(0, lambda n=name, c=confidence: self.grant_access(n, c))
                            
                            color = (0, 200, 80)  # Green
                        else:
                            # Check if we should log denied access
                            if name == "Unknown" and self.current_status == "scanning":
                                now = time.time()
                                if "Unknown" not in self.last_access or (now - self.last_access["Unknown"]) > Config.COOLDOWN_SECONDS:
                                    self.last_access["Unknown"] = now
                                    self.root.after(0, self.deny_access)
                            
                            color = (0, 0, 255)  # Red
                        
                        # Draw face box
                        cv2.rectangle(display_frame, (left, top), (right, bottom), color, 3)
                        
                        # Draw label
                        label = f"{name} ({confidence:.0%})" if name != "Unknown" else "Unknown"
                        cv2.rectangle(display_frame, (left, top - 40), (right, top), color, cv2.FILLED)
                        cv2.putText(display_frame, label, (left + 10, top - 10),
                                   cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                
                elif self.registration_mode:
                    # Registration mode - show face detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    
                    for (x, y, w, h) in faces:
                        cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 255), 3)
                    
                    # Show registration info
                    cv2.putText(display_frame, "REGISTRATION MODE", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(display_frame, f"Person: {self.registration_name}", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(display_frame, f"Captured: {self.captured_count}", (10, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Store current frame
                self.current_frame = frame.copy()
                
                # Update display
                self.root.after(0, lambda f=display_frame: self.display_frame(f))
                
                time.sleep(0.03)
            
        except Exception as e:
            print(f"Camera error: {e}")
        finally:
            self.camera.stop()
    
    def display_frame(self, frame):
        """Display a frame on the video label"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to fit display
        label_width = self.video_label.winfo_width()
        label_height = self.video_label.winfo_height()
        
        if label_width > 1 and label_height > 1:
            frame_h, frame_w = frame_rgb.shape[:2]
            scale = min(label_width / frame_w, label_height / frame_h)
            new_w = int(frame_w * scale)
            new_h = int(frame_h * scale)
            frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))
        
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
    
    def grant_access(self, name, confidence):
        """Handle access granted"""
        self.set_status("granted", name, confidence)
        self.access_log.add_entry(name, True, confidence)
        self.door_controller.unlock()
        self.update_log_display()
        print(f"[ACCESS] Granted: {name} ({confidence:.1%})")
    
    def deny_access(self):
        """Handle access denied"""
        self.set_status("denied")
        self.access_log.add_entry("Unknown", False, 0.0)
        self.update_log_display()
        print("[ACCESS] Denied: Unknown person")
    
    def show_admin_login(self):
        """Show admin login dialog"""
        password = simpledialog.askstring("Admin Login", "Enter admin password:", show='*')
        if password == Config.ADMIN_PASSWORD:
            self.show_admin_panel()
        elif password is not None:
            messagebox.showerror("Error", "Invalid password")
    
    def show_admin_panel(self):
        """Show the admin control panel"""
        self.admin_mode = True
        self.is_scanning = False
        
        # Create admin window
        self.admin_window = tk.Toplevel(self.root)
        self.admin_window.title("Admin Panel")
        self.admin_window.geometry("600x700")
        self.admin_window.configure(bg=Config.COLOR_DARK_BG)
        self.admin_window.transient(self.root)
        self.admin_window.grab_set()
        
        # Title
        tk.Label(
            self.admin_window,
            text="⚙ ADMIN PANEL",
            font=("Helvetica", 24, "bold"),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_DARK_BG
        ).pack(pady=20)
        
        # Notebook for tabs
        style = ttk.Style()
        style.configure('Admin.TNotebook', background=Config.COLOR_DARK_BG)
        style.configure('Admin.TFrame', background=Config.COLOR_DARK_BG)
        
        notebook = ttk.Notebook(self.admin_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Tab 1: Register New Face
        register_tab = tk.Frame(notebook, bg=Config.COLOR_DARK_BG)
        notebook.add(register_tab, text="📷 Register")
        self.create_register_tab(register_tab)
        
        # Tab 2: Train Model
        train_tab = tk.Frame(notebook, bg=Config.COLOR_DARK_BG)
        notebook.add(train_tab, text="🧠 Train")
        self.create_train_tab(train_tab)
        
        # Tab 3: Manage Users
        manage_tab = tk.Frame(notebook, bg=Config.COLOR_DARK_BG)
        notebook.add(manage_tab, text="👥 Manage")
        self.create_manage_tab(manage_tab)
        
        # Tab 4: Access Log
        log_tab = tk.Frame(notebook, bg=Config.COLOR_DARK_BG)
        notebook.add(log_tab, text="📋 Log")
        self.create_log_tab(log_tab)
        
        # Tab 5: Settings
        settings_tab = tk.Frame(notebook, bg=Config.COLOR_DARK_BG)
        notebook.add(settings_tab, text="⚙ Settings")
        self.create_settings_tab(settings_tab)
        
        # Close button
        close_btn = tk.Button(
            self.admin_window,
            text="Close Admin Panel",
            font=("Helvetica", 14),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_DENIED,
            activebackground="#cc0000",
            command=self.close_admin_panel
        )
        close_btn.pack(pady=20)
        
        self.admin_window.protocol("WM_DELETE_WINDOW", self.close_admin_panel)
    
    def create_register_tab(self, parent):
        """Create registration tab in admin panel"""
        tk.Label(
            parent,
            text="Register New Person",
            font=("Helvetica", 16, "bold"),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_DARK_BG
        ).pack(pady=20)
        
        # Name entry
        name_frame = tk.Frame(parent, bg=Config.COLOR_DARK_BG)
        name_frame.pack(fill=tk.X, padx=40, pady=10)
        
        tk.Label(
            name_frame,
            text="Name:",
            font=("Helvetica", 12),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_DARK_BG
        ).pack(side=tk.LEFT)
        
        self.reg_name_entry = tk.Entry(name_frame, font=("Helvetica", 12), width=30)
        self.reg_name_entry.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))
        
        # Capture count
        self.reg_count_label = tk.Label(
            parent,
            text="Photos captured: 0",
            font=("Helvetica", 12),
            fg=Config.COLOR_TEXT_DIM,
            bg=Config.COLOR_DARK_BG
        )
        self.reg_count_label.pack(pady=10)
        
        # Buttons
        btn_frame = tk.Frame(parent, bg=Config.COLOR_DARK_BG)
        btn_frame.pack(fill=tk.X, padx=40, pady=20)
        
        self.start_reg_btn = tk.Button(
            btn_frame,
            text="▶ Start Registration",
            font=("Helvetica", 12),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_GRANTED,
            activebackground="#00a040",
            command=self.start_registration
        )
        self.start_reg_btn.pack(fill=tk.X, pady=5)
        
        self.capture_btn = tk.Button(
            btn_frame,
            text="📸 Capture Photo",
            font=("Helvetica", 12),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_SCANNING,
            activebackground="#1976D2",
            command=self.capture_photo,
            state=tk.DISABLED
        )
        self.capture_btn.pack(fill=tk.X, pady=5)
        
        self.stop_reg_btn = tk.Button(
            btn_frame,
            text="⏹ Stop Registration",
            font=("Helvetica", 12),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_DENIED,
            activebackground="#cc0000",
            command=self.stop_registration,
            state=tk.DISABLED
        )
        self.stop_reg_btn.pack(fill=tk.X, pady=5)
        
        # Tips
        tk.Label(
            parent,
            text="Tips:\n• Capture 10-20 photos from different angles\n• Ensure good lighting\n• Look directly at camera",
            font=("Helvetica", 10),
            fg=Config.COLOR_TEXT_DIM,
            bg=Config.COLOR_DARK_BG,
            justify=tk.LEFT
        ).pack(pady=20)
    
    def create_train_tab(self, parent):
        """Create training tab in admin panel"""
        tk.Label(
            parent,
            text="Train Recognition Model",
            font=("Helvetica", 16, "bold"),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_DARK_BG
        ).pack(pady=20)
        
        # Dataset info
        info_frame = tk.Frame(parent, bg=Config.COLOR_PANEL_BG, bd=2, relief=tk.RAISED)
        info_frame.pack(fill=tk.X, padx=40, pady=10)
        
        persons = self.face_system.get_registered_persons()
        total_images = sum(count for _, count in persons)
        
        self.dataset_info_label = tk.Label(
            info_frame,
            text=f"Dataset: {len(persons)} persons, {total_images} images",
            font=("Helvetica", 12),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_PANEL_BG,
            pady=15
        )
        self.dataset_info_label.pack()
        
        # Progress bar
        progress_frame = tk.Frame(parent, bg=Config.COLOR_DARK_BG)
        progress_frame.pack(fill=tk.X, padx=40, pady=20)
        
        self.train_progress = ttk.Progressbar(progress_frame, mode='determinate', length=400)
        self.train_progress.pack(fill=tk.X)
        
        self.train_status_label = tk.Label(
            progress_frame,
            text="Ready to train",
            font=("Helvetica", 10),
            fg=Config.COLOR_TEXT_DIM,
            bg=Config.COLOR_DARK_BG
        )
        self.train_status_label.pack(pady=10)
        
        # Train button
        self.train_btn = tk.Button(
            parent,
            text="🧠 Start Training",
            font=("Helvetica", 14, "bold"),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_GRANTED,
            activebackground="#00a040",
            command=self.start_training,
            padx=40,
            pady=10
        )
        self.train_btn.pack(pady=20)
    
    def create_manage_tab(self, parent):
        """Create user management tab in admin panel"""
        tk.Label(
            parent,
            text="Manage Registered Persons",
            font=("Helvetica", 16, "bold"),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_DARK_BG
        ).pack(pady=20)
        
        # List of persons
        list_frame = tk.Frame(parent, bg=Config.COLOR_DARK_BG)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=10)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.manage_listbox = tk.Listbox(
            list_frame,
            font=("Helvetica", 12),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_PANEL_BG,
            selectbackground=Config.COLOR_SCANNING,
            yscrollcommand=scrollbar.set,
            height=10
        )
        self.manage_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.manage_listbox.yview)
        
        # Populate list
        self.refresh_manage_list()
        
        # Buttons
        btn_frame = tk.Frame(parent, bg=Config.COLOR_DARK_BG)
        btn_frame.pack(fill=tk.X, padx=40, pady=10)
        
        tk.Button(
            btn_frame,
            text="🔄 Refresh",
            font=("Helvetica", 11),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_SCANNING,
            command=self.refresh_manage_list
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Button(
            btn_frame,
            text="🗑 Delete Selected",
            font=("Helvetica", 11),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_DENIED,
            command=self.delete_person
        ).pack(side=tk.RIGHT, padx=5)
    
    def create_log_tab(self, parent):
        """Create access log tab in admin panel"""
        tk.Label(
            parent,
            text="Access Log",
            font=("Helvetica", 16, "bold"),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_DARK_BG
        ).pack(pady=20)
        
        # Log list
        list_frame = tk.Frame(parent, bg=Config.COLOR_DARK_BG)
        list_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=10)
        
        scrollbar = tk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.admin_log_listbox = tk.Listbox(
            list_frame,
            font=("Consolas", 11),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_PANEL_BG,
            selectbackground=Config.COLOR_SCANNING,
            yscrollcommand=scrollbar.set,
            height=15
        )
        self.admin_log_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.admin_log_listbox.yview)
        
        # Populate
        for entry in self.access_log.get_recent(50):
            timestamp = datetime.fromisoformat(entry['timestamp']).strftime("%Y-%m-%d %H:%M:%S")
            status = "GRANTED" if entry['access_granted'] else "DENIED"
            self.admin_log_listbox.insert(tk.END, f"{timestamp} | {status:7} | {entry['name']}")
        
        # Buttons
        btn_frame = tk.Frame(parent, bg=Config.COLOR_DARK_BG)
        btn_frame.pack(fill=tk.X, padx=40, pady=10)
        
        tk.Button(
            btn_frame,
            text="🗑 Clear Log",
            font=("Helvetica", 11),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_DENIED,
            command=self.clear_access_log
        ).pack(side=tk.RIGHT)
    
    def create_settings_tab(self, parent):
        """Create settings tab in admin panel"""
        tk.Label(
            parent,
            text="System Settings",
            font=("Helvetica", 16, "bold"),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_DARK_BG
        ).pack(pady=20)
        
        settings_frame = tk.Frame(parent, bg=Config.COLOR_DARK_BG)
        settings_frame.pack(fill=tk.X, padx=40)
        
        # Recognition threshold
        tk.Label(
            settings_frame,
            text=f"Recognition Threshold: {Config.RECOGNITION_THRESHOLD}",
            font=("Helvetica", 12),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_DARK_BG
        ).pack(anchor=tk.W, pady=5)
        
        # Cooldown
        tk.Label(
            settings_frame,
            text=f"Access Cooldown: {Config.COOLDOWN_SECONDS} seconds",
            font=("Helvetica", 12),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_DARK_BG
        ).pack(anchor=tk.W, pady=5)
        
        # Door unlock duration
        tk.Label(
            settings_frame,
            text=f"Door Unlock Duration: {Config.DOOR_UNLOCK_DURATION} seconds",
            font=("Helvetica", 12),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_DARK_BG
        ).pack(anchor=tk.W, pady=5)
        
        # System info
        tk.Label(
            settings_frame,
            text=f"\nSystem Info:",
            font=("Helvetica", 14, "bold"),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_DARK_BG
        ).pack(anchor=tk.W, pady=(20, 5))
        
        camera_type = "Raspberry Pi Camera" if USE_PICAMERA else "USB Webcam"
        gpio_status = "Available" if USE_GPIO else "Simulated"
        
        tk.Label(
            settings_frame,
            text=f"Camera: {camera_type}\nGPIO Control: {gpio_status}",
            font=("Helvetica", 12),
            fg=Config.COLOR_TEXT_DIM,
            bg=Config.COLOR_DARK_BG
        ).pack(anchor=tk.W, pady=5)
        
        # Exit kiosk button
        tk.Button(
            parent,
            text="🚪 Exit Kiosk",
            font=("Helvetica", 12),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_DENIED,
            command=self.exit_kiosk
        ).pack(pady=30)
    
    def refresh_manage_list(self):
        """Refresh the manage users list"""
        self.manage_listbox.delete(0, tk.END)
        persons = self.face_system.get_registered_persons()
        for name, count in persons:
            self.manage_listbox.insert(tk.END, f"{name} ({count} photos)")
    
    def start_registration(self):
        """Start face registration mode"""
        name = self.reg_name_entry.get().strip()
        if not name:
            messagebox.showwarning("Warning", "Please enter a person's name")
            return
        
        self.registration_mode = True
        self.registration_name = name
        self.captured_count = 0
        
        self.start_reg_btn.config(state=tk.DISABLED)
        self.capture_btn.config(state=tk.NORMAL)
        self.stop_reg_btn.config(state=tk.NORMAL)
        self.reg_name_entry.config(state=tk.DISABLED)
        
        self.reg_count_label.config(text=f"Photos captured: {self.captured_count}")
    
    def capture_photo(self):
        """Capture a photo for registration"""
        if self.current_frame is not None and self.registration_mode:
            filepath = self.face_system.save_face_image(self.current_frame, self.registration_name)
            self.captured_count += 1
            self.reg_count_label.config(text=f"Photos captured: {self.captured_count}")
            print(f"[REGISTER] Saved: {filepath}")
    
    def stop_registration(self):
        """Stop face registration mode"""
        self.registration_mode = False
        self.registration_name = ""
        
        self.start_reg_btn.config(state=tk.NORMAL)
        self.capture_btn.config(state=tk.DISABLED)
        self.stop_reg_btn.config(state=tk.DISABLED)
        self.reg_name_entry.config(state=tk.NORMAL)
        self.reg_name_entry.delete(0, tk.END)
        
        self.refresh_manage_list()
    
    def start_training(self):
        """Start model training"""
        self.train_btn.config(state=tk.DISABLED)
        self.train_progress['value'] = 0
        self.train_status_label.config(text="Training in progress...")
        
        def training_thread():
            def progress_callback(current, total, filepath):
                progress = (current / total) * 100
                self.root.after(0, lambda: self.train_progress.configure(value=progress))
                self.root.after(0, lambda: self.train_status_label.config(
                    text=f"Processing {current}/{total}..."))
            
            success, message = self.face_system.train_model(progress_callback)
            self.root.after(0, lambda: self.training_complete(success, message))
        
        thread = threading.Thread(target=training_thread, daemon=True)
        thread.start()
    
    def training_complete(self, success, message):
        """Handle training completion"""
        self.train_btn.config(state=tk.NORMAL)
        self.train_progress['value'] = 100 if success else 0
        self.train_status_label.config(text=message)
        
        if success:
            messagebox.showinfo("Training Complete", message)
            self.update_info_label()
        else:
            messagebox.showerror("Training Failed", message)
    
    def delete_person(self):
        """Delete selected person from dataset"""
        selection = self.manage_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a person to delete")
            return
        
        item = self.manage_listbox.get(selection[0])
        name = item.split(" (")[0]
        
        if messagebox.askyesno("Confirm Delete", f"Delete all photos for '{name}'?"):
            import shutil
            person_folder = os.path.join(self.face_system.dataset_path, name)
            if os.path.exists(person_folder):
                shutil.rmtree(person_folder)
                self.refresh_manage_list()
    
    def clear_access_log(self):
        """Clear the access log"""
        if messagebox.askyesno("Confirm", "Clear all access log entries?"):
            self.access_log.clear()
            self.admin_log_listbox.delete(0, tk.END)
            self.update_log_display()
    
    def update_info_label(self):
        """Update the info label"""
        count = len(self.face_system.get_trained_persons())
        self.info_label.config(text=f"Model: {count} persons registered | Press F1 for admin")
    
    def close_admin_panel(self):
        """Close the admin panel"""
        if self.registration_mode:
            self.stop_registration()
        
        self.admin_mode = False
        self.is_scanning = True
        self.admin_window.destroy()
        self.update_log_display()
        self.update_info_label()
    
    def exit_kiosk(self):
        """Exit the kiosk application"""
        if messagebox.askyesno("Exit", "Are you sure you want to exit the kiosk?"):
            self.on_closing()
    
    def on_closing(self):
        """Handle window close event"""
        self.is_running = False
        self.is_scanning = False
        
        if self.camera_thread:
            self.camera_thread.join(timeout=2)
        
        self.door_controller.cleanup()
        self.root.destroy()


# ==================== MAIN ====================
def main():
    """Main entry point"""
    root = tk.Tk()
    app = DoorEntryKiosk(root)
    root.mainloop()


if __name__ == "__main__":
    main()
