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
    
    # Performance Settings
    RECOGNITION_INTERVAL_FRAMES = 3  # Only run recognition every N frames
    FACE_CACHE_TTL = 2.0  # Seconds to cache a recognized face
    FACE_POSITION_TOLERANCE = 80  # Pixels tolerance for face position matching
    DETECTION_SCALE_FACTOR = 4  # Scale down factor for faster processing
    USE_FAST_DETECTION = True  # Use Haar cascade for initial detection
    
    # Door Control (GPIO Pin for Raspberry Pi)
    DOOR_RELAY_PIN = 17
    DOOR_UNLOCK_DURATION = 5  # Seconds to keep door unlocked
    
    # File Paths
    DATASET_PATH = "dataset"
    ENCODINGS_PATH = "encodings.pickle"
    ACCESS_LOG_PATH = "access_log.json"
    
    # Apple-like Clean Design Colors
    COLOR_GRANTED = "#34C759"    # Apple Green
    COLOR_DENIED = "#FF3B30"     # Apple Red
    COLOR_SCANNING = "#007AFF"   # Apple Blue
    COLOR_WARNING = "#FF9500"    # Apple Orange
    
    # Light Theme
    COLOR_BG = "#F2F2F7"         # Light gray background
    COLOR_CARD = "#FFFFFF"       # White cards
    COLOR_CARD_SECONDARY = "#F9F9F9"  # Slightly off-white
    COLOR_TEXT = "#1C1C1E"       # Near black text
    COLOR_TEXT_SECONDARY = "#8E8E93"  # Gray text
    COLOR_TEXT_TERTIARY = "#AEAEB2"   # Light gray text
    COLOR_BORDER = "#E5E5EA"     # Subtle border
    COLOR_SHADOW = "#C7C7CC"     # Shadow color
    
    # Typography (System fonts that look like SF Pro)
    FONT_FAMILY = "SF Pro Display" if os.name == 'darwin' else "Segoe UI" if os.name == 'nt' else "Helvetica Neue"
    FONT_FAMILY_MONO = "SF Mono" if os.name == 'darwin' else "Consolas" if os.name == 'nt' else "Monaco"


# ==================== FACE CACHE ====================
class FaceCache:
    """Caches recognized faces to avoid repeated recognition of the same face"""
    
    def __init__(self, ttl=None, position_tolerance=None):
        self.ttl = ttl or Config.FACE_CACHE_TTL
        self.position_tolerance = position_tolerance or Config.FACE_POSITION_TOLERANCE
        self.cache = {}  # {cache_key: {name, confidence, location, timestamp, encoding}}
        self.lock = threading.Lock()
    
    def _get_position_key(self, location):
        """Generate a grid-based position key for face location"""
        top, right, bottom, left = location
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2
        # Round to grid cells based on tolerance
        grid_x = center_x // self.position_tolerance
        grid_y = center_y // self.position_tolerance
        return (grid_x, grid_y)
    
    def _find_nearby_cache(self, location):
        """Find a cached face near the given location"""
        top, right, bottom, left = location
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2
        
        now = time.time()
        best_match = None
        best_distance = float('inf')
        
        with self.lock:
            # Clean expired entries and find closest match
            expired_keys = []
            for key, entry in self.cache.items():
                if now - entry['timestamp'] > self.ttl:
                    expired_keys.append(key)
                    continue
                
                cached_top, cached_right, cached_bottom, cached_left = entry['location']
                cached_center_x = (cached_left + cached_right) // 2
                cached_center_y = (cached_top + cached_bottom) // 2
                
                distance = ((center_x - cached_center_x) ** 2 + (center_y - cached_center_y) ** 2) ** 0.5
                
                if distance < self.position_tolerance and distance < best_distance:
                    best_distance = distance
                    best_match = entry
            
            # Remove expired entries
            for key in expired_keys:
                del self.cache[key]
        
        return best_match
    
    def get(self, location):
        """Get cached recognition result for a face at given location"""
        return self._find_nearby_cache(location)
    
    def put(self, location, name, confidence, encoding=None):
        """Cache a recognition result"""
        key = self._get_position_key(location)
        with self.lock:
            self.cache[key] = {
                'name': name,
                'confidence': confidence,
                'location': location,
                'timestamp': time.time(),
                'encoding': encoding
            }
    
    def clear(self):
        """Clear all cached entries"""
        with self.lock:
            self.cache.clear()
    
    def cleanup_expired(self):
        """Remove expired entries from cache"""
        now = time.time()
        with self.lock:
            expired_keys = [k for k, v in self.cache.items() if now - v['timestamp'] > self.ttl]
            for key in expired_keys:
                del self.cache[key]


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
    """Core face recognition logic with performance optimizations"""
    
    def __init__(self, dataset_path=None, encodings_path=None):
        self.dataset_path = dataset_path or Config.DATASET_PATH
        self.encodings_path = encodings_path or Config.ENCODINGS_PATH
        self.known_encodings = []
        self.known_names = []
        self.cv_scaler = Config.DETECTION_SCALE_FACTOR
        
        # Performance: Pre-load Haar cascade for fast face detection
        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Performance: Face cache to avoid repeated recognition
        self.face_cache = FaceCache()
        
        # Performance: Frame counter for skipping
        self.frame_count = 0
        
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
    
    def detect_faces_fast(self, frame):
        """Fast face detection using Haar cascade (no recognition)"""
        small_frame = cv2.resize(frame, (0, 0), fx=1/self.cv_scaler, fy=1/self.cv_scaler)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with Haar cascade (fast)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Convert to face_recognition format and scale up
        locations = []
        for (x, y, w, h) in faces:
            # Scale back to original frame size
            top = y * self.cv_scaler
            right = (x + w) * self.cv_scaler
            bottom = (y + h) * self.cv_scaler
            left = x * self.cv_scaler
            locations.append((top, right, bottom, left))
        
        return locations
    
    def recognize_faces(self, frame, force_recognition=False):
        """Detect and recognize faces in a frame with caching and optimization"""
        self.frame_count += 1
        
        if not self.known_encodings:
            return frame, []
        
        results = []
        
        # Step 1: Fast face detection using Haar cascade
        if Config.USE_FAST_DETECTION:
            fast_locations = self.detect_faces_fast(frame)
            
            # If no faces detected by fast method, skip expensive recognition
            if not fast_locations:
                return frame, []
        
        # Step 2: Check cache for each detected face
        faces_to_recognize = []
        cached_results = []
        
        if Config.USE_FAST_DETECTION:
            for location in fast_locations:
                cached = self.face_cache.get(location)
                if cached and not force_recognition:
                    # Use cached result
                    cached_results.append({
                        'name': cached['name'],
                        'confidence': cached['confidence'],
                        'location': location,
                        'from_cache': True
                    })
                else:
                    faces_to_recognize.append(location)
        
        # Step 3: Only run expensive face_recognition if needed
        should_recognize = (
            force_recognition or
            len(faces_to_recognize) > 0 or
            (self.frame_count % Config.RECOGNITION_INTERVAL_FRAMES == 0 and Config.USE_FAST_DETECTION)
        )
        
        if should_recognize and (faces_to_recognize or not Config.USE_FAST_DETECTION):
            # Prepare frame for face_recognition
            small_frame = cv2.resize(frame, (0, 0), fx=1/self.cv_scaler, fy=1/self.cv_scaler)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
            
            # Get face locations and encodings
            if Config.USE_FAST_DETECTION and faces_to_recognize:
                # Use the locations we already found, scaled down
                scaled_locations = [
                    (t // self.cv_scaler, r // self.cv_scaler, 
                     b // self.cv_scaler, l // self.cv_scaler)
                    for (t, r, b, l) in faces_to_recognize
                ]
                face_encodings = face_recognition.face_encodings(rgb_small, scaled_locations)
                face_locations = faces_to_recognize
            else:
                # Full detection with face_recognition library
                scaled_locations = face_recognition.face_locations(rgb_small)
                face_encodings = face_recognition.face_encodings(rgb_small, scaled_locations)
                # Scale up locations
                face_locations = [
                    (t * self.cv_scaler, r * self.cv_scaler,
                     b * self.cv_scaler, l * self.cv_scaler)
                    for (t, r, b, l) in scaled_locations
                ]
            
            # Recognize each face
            for location, face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(self.known_encodings, face_encoding)
                name = "Unknown"
                confidence = 0.0
                
                if True in matches:
                    face_distances = face_recognition.face_distance(self.known_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = self.known_names[best_match_index]
                        confidence = 1 - face_distances[best_match_index]
                
                # Cache this result
                self.face_cache.put(location, name, confidence, face_encoding)
                
                results.append({
                    'name': name,
                    'confidence': confidence,
                    'location': location,
                    'from_cache': False
                })
        
        # Combine cached and new results
        results.extend(cached_results)
        
        return frame, results
    
    def clear_cache(self):
        """Clear the face recognition cache"""
        self.face_cache.clear()


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
        
        self.root.configure(bg=Config.COLOR_BG)
        
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
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        self.faces_detected = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
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
        """Create the main kiosk interface with Apple-like design"""
        # Configure root window
        self.root.configure(bg=Config.COLOR_BG)
        
        # Main container with padding
        self.main_frame = tk.Frame(self.root, bg=Config.COLOR_BG)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=40, pady=30)
        
        # Header area
        header_frame = tk.Frame(self.main_frame, bg=Config.COLOR_BG)
        header_frame.pack(fill=tk.X, pady=(0, 30))
        
        # Title - clean, minimal
        self.title_label = tk.Label(
            header_frame, 
            text="Door Entry", 
            font=(Config.FONT_FAMILY, 34, "bold"),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_BG
        )
        self.title_label.pack(side=tk.LEFT)
        
        # Time display - right aligned, elegant
        time_frame = tk.Frame(header_frame, bg=Config.COLOR_BG)
        time_frame.pack(side=tk.RIGHT)
        
        self.time_label = tk.Label(
            time_frame,
            text="",
            font=(Config.FONT_FAMILY, 15),
            fg=Config.COLOR_TEXT_SECONDARY,
            bg=Config.COLOR_BG,
            justify=tk.RIGHT
        )
        self.time_label.pack()
        self.update_time()
        
        # Content area
        content_frame = tk.Frame(self.main_frame, bg=Config.COLOR_BG)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left side - Camera feed in a clean card
        camera_card = tk.Frame(content_frame, bg=Config.COLOR_CARD, highlightbackground=Config.COLOR_BORDER, highlightthickness=1)
        camera_card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 20))
        
        # Camera header
        camera_header = tk.Frame(camera_card, bg=Config.COLOR_CARD)
        camera_header.pack(fill=tk.X, padx=20, pady=(20, 10))
        
        tk.Label(
            camera_header,
            text="Camera",
            font=(Config.FONT_FAMILY, 13, "bold"),
            fg=Config.COLOR_TEXT_SECONDARY,
            bg=Config.COLOR_CARD
        ).pack(side=tk.LEFT)
        
        # FPS indicator
        self.fps_indicator = tk.Label(
            camera_header,
            text="● Live",
            font=(Config.FONT_FAMILY, 11),
            fg=Config.COLOR_GRANTED,
            bg=Config.COLOR_CARD
        )
        self.fps_indicator.pack(side=tk.RIGHT)
        
        # Video frame container - fixed aspect ratio
        video_container = tk.Frame(camera_card, bg=Config.COLOR_CARD_SECONDARY)
        video_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        video_container.pack_propagate(False)  # Prevent container from resizing to fit content
        
        self.video_label = tk.Label(video_container, bg=Config.COLOR_CARD_SECONDARY)
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Store reference to video container for sizing
        self.video_container = video_container
        
        # Right side - Status panel
        right_panel = tk.Frame(content_frame, bg=Config.COLOR_BG, width=380)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        right_panel.pack_propagate(False)
        
        # Status Card
        self.status_card = tk.Frame(right_panel, bg=Config.COLOR_CARD, highlightbackground=Config.COLOR_BORDER, highlightthickness=1)
        self.status_card.pack(fill=tk.X, pady=(0, 20))
        
        # Status content
        status_content = tk.Frame(self.status_card, bg=Config.COLOR_CARD)
        status_content.pack(fill=tk.X, padx=30, pady=40)
        
        self.status_icon_label = tk.Label(
            status_content,
            text="◉",
            font=(Config.FONT_FAMILY, 72),
            fg=Config.COLOR_SCANNING,
            bg=Config.COLOR_CARD
        )
        self.status_icon_label.pack()
        
        self.status_text_label = tk.Label(
            status_content,
            text="Ready to Scan",
            font=(Config.FONT_FAMILY, 22, "bold"),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_CARD
        )
        self.status_text_label.pack(pady=(15, 5))
        
        self.status_detail_label = tk.Label(
            status_content,
            text="Position your face in the camera",
            font=(Config.FONT_FAMILY, 13),
            fg=Config.COLOR_TEXT_SECONDARY,
            bg=Config.COLOR_CARD
        )
        self.status_detail_label.pack()
        
        # Store status frame reference for background updates
        self.status_frame = status_content
        
        # Recent Activity Card
        activity_card = tk.Frame(right_panel, bg=Config.COLOR_CARD, highlightbackground=Config.COLOR_BORDER, highlightthickness=1)
        activity_card.pack(fill=tk.BOTH, expand=True)
        
        # Activity header
        activity_header = tk.Frame(activity_card, bg=Config.COLOR_CARD)
        activity_header.pack(fill=tk.X, padx=20, pady=(20, 15))
        
        tk.Label(
            activity_header,
            text="Recent Activity",
            font=(Config.FONT_FAMILY, 13, "bold"),
            fg=Config.COLOR_TEXT_SECONDARY,
            bg=Config.COLOR_CARD
        ).pack(side=tk.LEFT)
        
        # Activity list container
        list_container = tk.Frame(activity_card, bg=Config.COLOR_CARD)
        list_container.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        self.log_listbox = tk.Listbox(
            list_container,
            font=(Config.FONT_FAMILY_MONO, 12),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_CARD,
            selectbackground=Config.COLOR_BG,
            selectforeground=Config.COLOR_TEXT,
            highlightthickness=0,
            bd=0,
            relief=tk.FLAT,
            activestyle='none'
        )
        self.log_listbox.pack(fill=tk.BOTH, expand=True)
        
        # Footer
        footer_frame = tk.Frame(self.main_frame, bg=Config.COLOR_BG)
        footer_frame.pack(fill=tk.X, pady=(30, 0))
        
        # Admin button - subtle, clean
        self.admin_btn = tk.Button(
            footer_frame,
            text="Settings",
            font=(Config.FONT_FAMILY, 12),
            fg=Config.COLOR_TEXT_SECONDARY,
            bg=Config.COLOR_BG,
            activebackground=Config.COLOR_BG,
            activeforeground=Config.COLOR_SCANNING,
            bd=0,
            cursor="hand2",
            command=self.show_admin_login
        )
        self.admin_btn.pack(side=tk.LEFT)
        
        # Status info
        self.info_label = tk.Label(
            footer_frame,
            text=f"{len(self.face_system.get_trained_persons())} registered users",
            font=(Config.FONT_FAMILY, 11),
            fg=Config.COLOR_TEXT_TERTIARY,
            bg=Config.COLOR_BG
        )
        self.info_label.pack(side=tk.RIGHT)
        
        # Load recent log entries
        self.update_log_display()
    
    def update_time(self):
        """Update the time display"""
        current_time = datetime.now().strftime("%H:%M")
        current_date = datetime.now().strftime("%A, %B %d")
        self.time_label.config(text=f"{current_date}  •  {current_time}")
        self.root.after(1000, self.update_time)
    
    def update_log_display(self):
        """Update the access log display"""
        self.log_listbox.delete(0, tk.END)
        entries = self.access_log.get_recent(8)
        
        for entry in entries:
            timestamp = datetime.fromisoformat(entry['timestamp']).strftime("%H:%M")
            status_icon = "●" if entry['access_granted'] else "○"
            status_color = "" 
            name = entry['name'][:15] + "..." if len(entry['name']) > 15 else entry['name']
            self.log_listbox.insert(tk.END, f"  {status_icon}  {timestamp}   {name}")
    
    def set_status(self, status, name="", confidence=0.0):
        """Update the status display with Apple-like styling"""
        self.current_status = status
        
        if status == "granted":
            self.status_icon_label.config(text="✓", fg=Config.COLOR_GRANTED)
            self.status_text_label.config(text="Welcome", fg=Config.COLOR_TEXT)
            self.status_detail_label.config(text=f"{name}")
            self.status_card.config(bg=Config.COLOR_GRANTED, highlightbackground=Config.COLOR_GRANTED)
            for widget in self.status_frame.winfo_children():
                widget.config(bg=Config.COLOR_GRANTED)
                if widget == self.status_icon_label:
                    widget.config(fg="#FFFFFF")
                elif widget in [self.status_text_label, self.status_detail_label]:
                    widget.config(fg="#FFFFFF")
            self.status_frame.config(bg=Config.COLOR_GRANTED)
            self.root.after(3000, lambda: self.set_status("scanning"))
            
        elif status == "denied":
            self.status_icon_label.config(text="✕", fg=Config.COLOR_DENIED)
            self.status_text_label.config(text="Not Recognized", fg=Config.COLOR_TEXT)
            self.status_detail_label.config(text="Access denied")
            self.status_card.config(bg=Config.COLOR_DENIED, highlightbackground=Config.COLOR_DENIED)
            for widget in self.status_frame.winfo_children():
                widget.config(bg=Config.COLOR_DENIED)
                if widget == self.status_icon_label:
                    widget.config(fg="#FFFFFF")
                elif widget in [self.status_text_label, self.status_detail_label]:
                    widget.config(fg="#FFFFFF")
            self.status_frame.config(bg=Config.COLOR_DENIED)
            self.root.after(3000, lambda: self.set_status("scanning"))
            
        else:  # scanning
            self.status_icon_label.config(text="◉", fg=Config.COLOR_SCANNING)
            self.status_text_label.config(text="Ready to Scan", fg=Config.COLOR_TEXT)
            self.status_detail_label.config(text="Position your face in the camera")
            self.status_card.config(bg=Config.COLOR_CARD, highlightbackground=Config.COLOR_BORDER)
            for widget in self.status_frame.winfo_children():
                widget.config(bg=Config.COLOR_CARD)
            self.status_frame.config(bg=Config.COLOR_CARD)
            self.status_text_label.config(fg=Config.COLOR_TEXT)
            self.status_detail_label.config(fg=Config.COLOR_TEXT_SECONDARY)
    
    def start_camera(self):
        """Start the camera in a background thread"""
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()
    
    def camera_loop(self):
        """Main camera loop running in a separate thread"""
        try:
            self.camera.start()
            frame_time = time.time()
            
            while self.is_running:
                loop_start = time.time()
                
                frame = self.camera.capture_frame()
                if frame is None:
                    continue
                
                display_frame = frame.copy()
                
                if self.is_scanning and not self.registration_mode:
                    # Perform optimized face recognition
                    _, results = self.face_system.recognize_faces(frame)
                    
                    # Track performance metrics
                    self.faces_detected = len(results)
                    cache_hits_this_frame = sum(1 for r in results if r.get('from_cache', False))
                    cache_misses_this_frame = len(results) - cache_hits_this_frame
                    self.cache_hits += cache_hits_this_frame
                    self.cache_misses += cache_misses_this_frame
                    
                    for result in results:
                        top, right, bottom, left = result['location']
                        name = result['name']
                        confidence = result['confidence']
                        from_cache = result.get('from_cache', False)
                        
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
                            # Check if we should log denied access (only for non-cached results)
                            if name == "Unknown" and self.current_status == "scanning" and not from_cache:
                                now = time.time()
                                if "Unknown" not in self.last_access or (now - self.last_access["Unknown"]) > Config.COOLDOWN_SECONDS:
                                    self.last_access["Unknown"] = now
                                    self.root.after(0, self.deny_access)
                            
                            color = (0, 0, 255)  # Red
                        
                        # Draw face box (thinner for cached results)
                        box_thickness = 2 if from_cache else 3
                        cv2.rectangle(display_frame, (left, top), (right, bottom), color, box_thickness)
                        
                        # Draw label with cache indicator
                        cache_indicator = " [C]" if from_cache else ""
                        label = f"{name} ({confidence:.0%}){cache_indicator}" if name != "Unknown" else f"Unknown{cache_indicator}"
                        cv2.rectangle(display_frame, (left, top - 40), (right, top), color, cv2.FILLED)
                        cv2.putText(display_frame, label, (left + 10, top - 10),
                                   cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)
                
                elif self.registration_mode:
                    # Registration mode - use the pre-loaded cascade from face_system
                    faces = self.face_system.detect_faces_fast(frame)
                    
                    for (top, right, bottom, left) in faces:
                        cv2.rectangle(display_frame, (left, top), (right, bottom), (0, 255, 255), 3)
                    
                    # Show registration info
                    cv2.putText(display_frame, "REGISTRATION MODE", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.putText(display_frame, f"Person: {self.registration_name}", (10, 70),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(display_frame, f"Captured: {self.captured_count}", (10, 110),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(display_frame, f"Faces detected: {len(faces)}", (10, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
                
                # Calculate and display FPS
                self.fps_counter += 1
                elapsed = time.time() - self.fps_start_time
                if elapsed >= 1.0:
                    self.current_fps = self.fps_counter / elapsed
                    self.fps_counter = 0
                    self.fps_start_time = time.time()
                
                # Draw performance overlay
                fps_text = f"FPS: {self.current_fps:.1f}"
                cv2.putText(display_frame, fps_text, (display_frame.shape[1] - 120, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Store current frame
                self.current_frame = frame.copy()
                
                # Update display
                self.root.after(0, lambda f=display_frame: self.display_frame(f))
                
                # Adaptive frame rate - aim for ~30 FPS
                loop_time = time.time() - loop_start
                sleep_time = max(0.001, 0.033 - loop_time)
                time.sleep(sleep_time)
            
        except Exception as e:
            print(f"Camera error: {e}")
        finally:
            self.camera.stop()
    
    def display_frame(self, frame):
        """Display a frame on the video label"""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Get container size (not label size to avoid feedback loop)
        container_width = self.video_container.winfo_width()
        container_height = self.video_container.winfo_height()
        
        # Only resize if container has valid dimensions
        if container_width > 10 and container_height > 10:
            frame_h, frame_w = frame_rgb.shape[:2]
            
            # Calculate scale to fit within container while maintaining aspect ratio
            scale = min((container_width - 4) / frame_w, (container_height - 4) / frame_h)
            
            # Don't scale up beyond original size
            scale = min(scale, 1.5)
            
            new_w = int(frame_w * scale)
            new_h = int(frame_h * scale)
            
            # Ensure minimum size
            new_w = max(new_w, 320)
            new_h = max(new_h, 240)
            
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
        
        # Clear the face cache when entering admin mode
        self.face_system.clear_cache()
        
        # Create admin window with Apple-like styling
        self.admin_window = tk.Toplevel(self.root)
        self.admin_window.title("Settings")
        self.admin_window.geometry("650x750")
        self.admin_window.configure(bg=Config.COLOR_BG)
        self.admin_window.transient(self.root)
        self.admin_window.grab_set()
        
        # Title
        header = tk.Frame(self.admin_window, bg=Config.COLOR_BG)
        header.pack(fill=tk.X, padx=30, pady=(25, 15))
        
        tk.Label(
            header,
            text="Settings",
            font=(Config.FONT_FAMILY, 28, "bold"),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_BG
        ).pack(side=tk.LEFT)
        
        # Close button in header
        close_btn = tk.Button(
            header,
            text="Done",
            font=(Config.FONT_FAMILY, 14),
            fg=Config.COLOR_SCANNING,
            bg=Config.COLOR_BG,
            activeforeground=Config.COLOR_SCANNING,
            activebackground=Config.COLOR_BG,
            bd=0,
            cursor="hand2",
            command=self.close_admin_panel
        )
        close_btn.pack(side=tk.RIGHT)
        
        # Configure notebook style for Apple look
        style = ttk.Style()
        style.configure('TNotebook', background=Config.COLOR_BG, borderwidth=0)
        style.configure('TNotebook.Tab', 
                       font=(Config.FONT_FAMILY, 11),
                       padding=[20, 10],
                       background=Config.COLOR_BG,
                       foreground=Config.COLOR_TEXT_SECONDARY)
        style.map('TNotebook.Tab',
                 background=[('selected', Config.COLOR_BG)],
                 foreground=[('selected', Config.COLOR_SCANNING)])
        
        notebook = ttk.Notebook(self.admin_window)
        notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Tab 1: Register New Face
        register_tab = tk.Frame(notebook, bg=Config.COLOR_BG)
        notebook.add(register_tab, text="  Register  ")
        self.create_register_tab(register_tab)
        
        # Tab 2: Train Model
        train_tab = tk.Frame(notebook, bg=Config.COLOR_BG)
        notebook.add(train_tab, text="  Train  ")
        self.create_train_tab(train_tab)
        
        # Tab 3: Manage Users
        manage_tab = tk.Frame(notebook, bg=Config.COLOR_BG)
        notebook.add(manage_tab, text="  Users  ")
        self.create_manage_tab(manage_tab)
        
        # Tab 4: Access Log
        log_tab = tk.Frame(notebook, bg=Config.COLOR_BG)
        notebook.add(log_tab, text="  Activity  ")
        self.create_log_tab(log_tab)
        
        # Tab 5: Settings
        settings_tab = tk.Frame(notebook, bg=Config.COLOR_BG)
        notebook.add(settings_tab, text="  System  ")
        self.create_settings_tab(settings_tab)
        
        self.admin_window.protocol("WM_DELETE_WINDOW", self.close_admin_panel)
    
    def create_register_tab(self, parent):
        """Create registration tab in admin panel with Apple styling"""
        # Card container
        card = tk.Frame(parent, bg=Config.COLOR_CARD, highlightbackground=Config.COLOR_BORDER, highlightthickness=1)
        card.pack(fill=tk.X, padx=20, pady=20)
        
        inner = tk.Frame(card, bg=Config.COLOR_CARD)
        inner.pack(fill=tk.X, padx=25, pady=25)
        
        tk.Label(
            inner,
            text="Add New Person",
            font=(Config.FONT_FAMILY, 17, "bold"),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_CARD
        ).pack(anchor=tk.W)
        
        tk.Label(
            inner,
            text="Capture face photos for recognition training",
            font=(Config.FONT_FAMILY, 12),
            fg=Config.COLOR_TEXT_SECONDARY,
            bg=Config.COLOR_CARD
        ).pack(anchor=tk.W, pady=(5, 20))
        
        # Name entry
        tk.Label(
            inner,
            text="Full Name",
            font=(Config.FONT_FAMILY, 11),
            fg=Config.COLOR_TEXT_SECONDARY,
            bg=Config.COLOR_CARD
        ).pack(anchor=tk.W)
        
        self.reg_name_entry = tk.Entry(
            inner, 
            font=(Config.FONT_FAMILY, 14), 
            bg=Config.COLOR_CARD_SECONDARY,
            fg=Config.COLOR_TEXT,
            relief=tk.FLAT,
            highlightbackground=Config.COLOR_BORDER,
            highlightthickness=1
        )
        self.reg_name_entry.pack(fill=tk.X, pady=(5, 20), ipady=8)
        
        # Capture count
        self.reg_count_label = tk.Label(
            inner,
            text="0 photos captured",
            font=(Config.FONT_FAMILY, 13),
            fg=Config.COLOR_TEXT_SECONDARY,
            bg=Config.COLOR_CARD
        )
        self.reg_count_label.pack(pady=(0, 20))
        
        # Buttons frame
        btn_frame = tk.Frame(inner, bg=Config.COLOR_CARD)
        btn_frame.pack(fill=tk.X)
        
        self.start_reg_btn = tk.Button(
            btn_frame,
            text="Start Camera",
            font=(Config.FONT_FAMILY, 13),
            fg="#FFFFFF",
            bg=Config.COLOR_SCANNING,
            activebackground="#0056b3",
            activeforeground="#FFFFFF",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.start_registration
        )
        self.start_reg_btn.pack(fill=tk.X, pady=3, ipady=8)
        
        self.capture_btn = tk.Button(
            btn_frame,
            text="Capture Photo",
            font=(Config.FONT_FAMILY, 13),
            fg="#FFFFFF",
            bg=Config.COLOR_GRANTED,
            activebackground="#28a745",
            activeforeground="#FFFFFF",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.capture_photo,
            state=tk.DISABLED
        )
        self.capture_btn.pack(fill=tk.X, pady=3, ipady=8)
        
        self.stop_reg_btn = tk.Button(
            btn_frame,
            text="Stop",
            font=(Config.FONT_FAMILY, 13),
            fg=Config.COLOR_DENIED,
            bg=Config.COLOR_CARD,
            activeforeground=Config.COLOR_DENIED,
            activebackground=Config.COLOR_CARD_SECONDARY,
            relief=tk.FLAT,
            cursor="hand2",
            command=self.stop_registration,
            state=tk.DISABLED
        )
        self.stop_reg_btn.pack(fill=tk.X, pady=3, ipady=8)
        
        # Tips
        tips_frame = tk.Frame(parent, bg=Config.COLOR_BG)
        tips_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(
            tips_frame,
            text="Tips for best results",
            font=(Config.FONT_FAMILY, 11, "bold"),
            fg=Config.COLOR_TEXT_SECONDARY,
            bg=Config.COLOR_BG
        ).pack(anchor=tk.W)
        
        tk.Label(
            tips_frame,
            text="• Capture 10-20 photos from different angles\n• Ensure good, even lighting\n• Look directly at the camera",
            font=(Config.FONT_FAMILY, 11),
            fg=Config.COLOR_TEXT_TERTIARY,
            bg=Config.COLOR_BG,
            justify=tk.LEFT
        ).pack(anchor=tk.W, pady=(5, 0))
    
    def create_train_tab(self, parent):
        """Create training tab in admin panel with Apple styling"""
        # Card container
        card = tk.Frame(parent, bg=Config.COLOR_CARD, highlightbackground=Config.COLOR_BORDER, highlightthickness=1)
        card.pack(fill=tk.X, padx=20, pady=20)
        
        inner = tk.Frame(card, bg=Config.COLOR_CARD)
        inner.pack(fill=tk.X, padx=25, pady=25)
        
        tk.Label(
            inner,
            text="Train Model",
            font=(Config.FONT_FAMILY, 17, "bold"),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_CARD
        ).pack(anchor=tk.W)
        
        tk.Label(
            inner,
            text="Process captured photos to train the recognition model",
            font=(Config.FONT_FAMILY, 12),
            fg=Config.COLOR_TEXT_SECONDARY,
            bg=Config.COLOR_CARD
        ).pack(anchor=tk.W, pady=(5, 25))
        
        # Dataset info
        persons = self.face_system.get_registered_persons()
        total_images = sum(count for _, count in persons)
        
        info_card = tk.Frame(inner, bg=Config.COLOR_CARD_SECONDARY)
        info_card.pack(fill=tk.X, pady=(0, 20))
        
        self.dataset_info_label = tk.Label(
            info_card,
            text=f"{len(persons)} people  •  {total_images} photos",
            font=(Config.FONT_FAMILY, 14),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_CARD_SECONDARY,
            pady=15
        )
        self.dataset_info_label.pack()
        
        # Progress bar
        style = ttk.Style()
        style.configure("Custom.Horizontal.TProgressbar",
                       background=Config.COLOR_SCANNING,
                       troughcolor=Config.COLOR_CARD_SECONDARY)
        
        self.train_progress = ttk.Progressbar(
            inner, 
            mode='determinate', 
            length=400,
            style="Custom.Horizontal.TProgressbar"
        )
        self.train_progress.pack(fill=tk.X, pady=(0, 10))
        
        self.train_status_label = tk.Label(
            inner,
            text="Ready to train",
            font=(Config.FONT_FAMILY, 11),
            fg=Config.COLOR_TEXT_SECONDARY,
            bg=Config.COLOR_CARD
        )
        self.train_status_label.pack(pady=(0, 20))
        
        # Train button
        self.train_btn = tk.Button(
            inner,
            text="Start Training",
            font=(Config.FONT_FAMILY, 14),
            fg="#FFFFFF",
            bg=Config.COLOR_SCANNING,
            activebackground="#0056b3",
            activeforeground="#FFFFFF",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.start_training
        )
        self.train_btn.pack(fill=tk.X, ipady=10)
    
    def create_manage_tab(self, parent):
        """Create user management tab in admin panel with Apple styling"""
        # Header
        header = tk.Frame(parent, bg=Config.COLOR_BG)
        header.pack(fill=tk.X, padx=20, pady=(20, 10))
        
        tk.Label(
            header,
            text="Registered Users",
            font=(Config.FONT_FAMILY, 13, "bold"),
            fg=Config.COLOR_TEXT_SECONDARY,
            bg=Config.COLOR_BG
        ).pack(side=tk.LEFT)
        
        tk.Button(
            header,
            text="Refresh",
            font=(Config.FONT_FAMILY, 11),
            fg=Config.COLOR_SCANNING,
            bg=Config.COLOR_BG,
            activeforeground=Config.COLOR_SCANNING,
            bd=0,
            cursor="hand2",
            command=self.refresh_manage_list
        ).pack(side=tk.RIGHT)
        
        # List card
        card = tk.Frame(parent, bg=Config.COLOR_CARD, highlightbackground=Config.COLOR_BORDER, highlightthickness=1)
        card.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.manage_listbox = tk.Listbox(
            card,
            font=(Config.FONT_FAMILY, 13),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_CARD,
            selectbackground=Config.COLOR_CARD_SECONDARY,
            selectforeground=Config.COLOR_TEXT,
            highlightthickness=0,
            bd=0,
            relief=tk.FLAT,
            activestyle='none'
        )
        self.manage_listbox.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Populate list
        self.refresh_manage_list()
        
        # Delete button
        btn_frame = tk.Frame(parent, bg=Config.COLOR_BG)
        btn_frame.pack(fill=tk.X, padx=20, pady=15)
        
        tk.Button(
            btn_frame,
            text="Delete Selected",
            font=(Config.FONT_FAMILY, 12),
            fg=Config.COLOR_DENIED,
            bg=Config.COLOR_BG,
            activeforeground=Config.COLOR_DENIED,
            bd=0,
            cursor="hand2",
            command=self.delete_person
        ).pack(side=tk.RIGHT)
    
    def create_log_tab(self, parent):
        """Create access log tab in admin panel with Apple styling"""
        # Header
        header = tk.Frame(parent, bg=Config.COLOR_BG)
        header.pack(fill=tk.X, padx=20, pady=(20, 10))
        
        tk.Label(
            header,
            text="Access History",
            font=(Config.FONT_FAMILY, 13, "bold"),
            fg=Config.COLOR_TEXT_SECONDARY,
            bg=Config.COLOR_BG
        ).pack(side=tk.LEFT)
        
        tk.Button(
            header,
            text="Clear All",
            font=(Config.FONT_FAMILY, 11),
            fg=Config.COLOR_DENIED,
            bg=Config.COLOR_BG,
            activeforeground=Config.COLOR_DENIED,
            bd=0,
            cursor="hand2",
            command=self.clear_access_log
        ).pack(side=tk.RIGHT)
        
        # Log card
        card = tk.Frame(parent, bg=Config.COLOR_CARD, highlightbackground=Config.COLOR_BORDER, highlightthickness=1)
        card.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        self.admin_log_listbox = tk.Listbox(
            card,
            font=(Config.FONT_FAMILY_MONO, 11),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_CARD,
            selectbackground=Config.COLOR_CARD_SECONDARY,
            selectforeground=Config.COLOR_TEXT,
            highlightthickness=0,
            bd=0,
            relief=tk.FLAT,
            activestyle='none'
        )
        self.admin_log_listbox.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)
        
        # Populate
        for entry in self.access_log.get_recent(50):
            timestamp = datetime.fromisoformat(entry['timestamp']).strftime("%b %d, %H:%M")
            status_icon = "●" if entry['access_granted'] else "○"
            self.admin_log_listbox.insert(tk.END, f"  {status_icon}  {timestamp}    {entry['name']}")
    
    def create_settings_tab(self, parent):
        """Create settings tab in admin panel with Apple styling"""
        # Recognition Settings Card
        card1 = tk.Frame(parent, bg=Config.COLOR_CARD, highlightbackground=Config.COLOR_BORDER, highlightthickness=1)
        card1.pack(fill=tk.X, padx=20, pady=(20, 10))
        
        inner1 = tk.Frame(card1, bg=Config.COLOR_CARD)
        inner1.pack(fill=tk.X, padx=20, pady=20)
        
        tk.Label(
            inner1,
            text="Recognition",
            font=(Config.FONT_FAMILY, 15, "bold"),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_CARD
        ).pack(anchor=tk.W)
        
        settings_items = [
            ("Confidence Threshold", f"{int(Config.RECOGNITION_THRESHOLD * 100)}%"),
            ("Access Cooldown", f"{Config.COOLDOWN_SECONDS} seconds"),
            ("Door Unlock Duration", f"{Config.DOOR_UNLOCK_DURATION} seconds"),
        ]
        
        for label, value in settings_items:
            row = tk.Frame(inner1, bg=Config.COLOR_CARD)
            row.pack(fill=tk.X, pady=8)
            tk.Label(
                row,
                text=label,
                font=(Config.FONT_FAMILY, 13),
                fg=Config.COLOR_TEXT,
                bg=Config.COLOR_CARD
            ).pack(side=tk.LEFT)
            tk.Label(
                row,
                text=value,
                font=(Config.FONT_FAMILY, 13),
                fg=Config.COLOR_TEXT_SECONDARY,
                bg=Config.COLOR_CARD
            ).pack(side=tk.RIGHT)
        
        # System Info Card
        card2 = tk.Frame(parent, bg=Config.COLOR_CARD, highlightbackground=Config.COLOR_BORDER, highlightthickness=1)
        card2.pack(fill=tk.X, padx=20, pady=10)
        
        inner2 = tk.Frame(card2, bg=Config.COLOR_CARD)
        inner2.pack(fill=tk.X, padx=20, pady=20)
        
        tk.Label(
            inner2,
            text="System",
            font=(Config.FONT_FAMILY, 15, "bold"),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_CARD
        ).pack(anchor=tk.W)
        
        camera_type = "Raspberry Pi Camera" if USE_PICAMERA else "USB Webcam"
        gpio_status = "Hardware" if USE_GPIO else "Simulated"
        
        system_items = [
            ("Camera", camera_type),
            ("Door Control", gpio_status),
            ("Performance Mode", "Optimized" if Config.USE_FAST_DETECTION else "Standard"),
        ]
        
        for label, value in system_items:
            row = tk.Frame(inner2, bg=Config.COLOR_CARD)
            row.pack(fill=tk.X, pady=8)
            tk.Label(
                row,
                text=label,
                font=(Config.FONT_FAMILY, 13),
                fg=Config.COLOR_TEXT,
                bg=Config.COLOR_CARD
            ).pack(side=tk.LEFT)
            tk.Label(
                row,
                text=value,
                font=(Config.FONT_FAMILY, 13),
                fg=Config.COLOR_TEXT_SECONDARY,
                bg=Config.COLOR_CARD
            ).pack(side=tk.RIGHT)
        
        # Exit button
        exit_frame = tk.Frame(parent, bg=Config.COLOR_BG)
        exit_frame.pack(fill=tk.X, padx=20, pady=30)
        
        tk.Button(
            exit_frame,
            text="Exit Kiosk Mode",
            font=(Config.FONT_FAMILY, 13),
            fg=Config.COLOR_DENIED,
            bg=Config.COLOR_BG,
            activeforeground=Config.COLOR_DENIED,
            bd=0,
            cursor="hand2",
            command=self.exit_kiosk
        ).pack()
    
    def refresh_manage_list(self):
        """Refresh the manage users list"""
        self.manage_listbox.delete(0, tk.END)
        persons = self.face_system.get_registered_persons()
        for name, count in persons:
            self.manage_listbox.insert(tk.END, f"  {name}   •   {count} photos")
    
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
        
        self.reg_count_label.config(text=f"{self.captured_count} photos captured")
    
    def capture_photo(self):
        """Capture a photo for registration"""
        if self.current_frame is not None and self.registration_mode:
            filepath = self.face_system.save_face_image(self.current_frame, self.registration_name)
            self.captured_count += 1
            self.reg_count_label.config(text=f"{self.captured_count} photos captured")
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
        self.info_label.config(text=f"{count} registered users")
    
    def close_admin_panel(self):
        """Close the admin panel"""
        if self.registration_mode:
            self.stop_registration()
        
        self.admin_mode = False
        self.is_scanning = True
        
        # Clear cache and reset performance stats when returning to scanning
        self.face_system.clear_cache()
        self.cache_hits = 0
        self.cache_misses = 0
        
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
