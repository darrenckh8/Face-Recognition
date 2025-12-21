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
    RECOGNITION_THRESHOLD = 0.8  # Lower = more strict (0.0 - 1.0)
    COOLDOWN_SECONDS = 5  # Prevent repeated access logs for same person
    
    # Performance Settings
    RECOGNITION_INTERVAL_FRAMES = 3  # Only run recognition every N frames
    FACE_CACHE_TTL = 2.0  # Seconds to cache a recognized face
    FACE_POSITION_TOLERANCE = 80  # Pixels tolerance for face position matching
    DETECTION_SCALE_FACTOR = 4  # Scale down factor for faster processing
    USE_FAST_DETECTION = True  # Use Haar cascade for initial detection
    
    # Door Control (GPIO Pin for Raspberry Pi)
    DOOR_RELAY_PIN = 17
    DOOR_UNLOCK_DURATION = 1  # Seconds to keep door unlocked
    
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
        """Remove expired entries from cache, returns count of removed entries"""
        now = time.time()
        with self.lock:
            expired_keys = [k for k, v in self.cache.items() if now - v['timestamp'] > self.ttl]
            for key in expired_keys:
                del self.cache[key]
            return len(expired_keys)


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
    
    def get_filtered(self, date_from=None, date_to=None, name_filter=None, count=100):
        """Get entries filtered by date range and/or name"""
        filtered = []
        for entry in reversed(self.entries):
            # Parse entry timestamp
            entry_date = datetime.fromisoformat(entry['timestamp']).date()
            
            # Date range filter
            if date_from and entry_date < date_from:
                continue
            if date_to and entry_date > date_to:
                continue
            
            # Name filter (case-insensitive partial match)
            if name_filter and name_filter.lower() not in entry['name'].lower():
                continue
            
            filtered.append(entry)
            if len(filtered) >= count:
                break
        
        return filtered
    
    def get_unique_names(self):
        """Get list of unique names in the log"""
        names = set()
        for entry in self.entries:
            names.add(entry['name'])
        return sorted(list(names))
    
    def clear(self):
        """Clear all entries"""
        self.entries = []
        self.save()


# ==================== CAMERA MANAGER ====================
class CameraManager:
    """Manages camera operations in a separate thread for zero-latency frame capture"""
    
    def __init__(self, use_picamera=False, resolution=(640, 480)):
        self.use_picamera = use_picamera
        self.resolution = resolution
        self.camera = None
        self.is_running = False
        
        # Thread-safe frame storage
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.capture_thread = None
    
    def start(self):
        """Initialize and start the camera with background capture thread"""
        if self.use_picamera:
            self.camera = Picamera2()
            self.camera.configure(self.camera.create_preview_configuration(
                main={"format": 'XRGB8888', "size": self.resolution}, buffer_count=2
            ))
            self.camera.start()
        else:
            # Try multiple video indices in case camera mounts at /dev/video1 or /dev/video2
            camera_indices = [0, 1, 2]
            self.camera = None
            
            for idx in camera_indices:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    # Test if we can actually read a frame
                    ret, _ = cap.read()
                    if ret:
                        self.camera = cap
                        print(f"[CAMERA] Connected to video device {idx}")
                        break
                    else:
                        cap.release()
                else:
                    cap.release()
            
            if self.camera is None:
                raise RuntimeError("Could not connect to any camera. Tried indices: 0, 1, 2")
            
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            # Reduce buffer size to minimize latency
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        self.is_running = True
        
        # Start background capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        # Wait for first frame
        time.sleep(0.3)
    
    def _capture_loop(self):
        """Background thread that continuously captures frames"""
        while self.is_running:
            try:
                if self.use_picamera:
                    frame = self.camera.capture_array()
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                else:
                    ret, frame = self.camera.read()
                    if not ret:
                        continue
                
                # Store the latest frame thread-safely
                with self.frame_lock:
                    self.current_frame = frame
                    
            except Exception as e:
                print(f"[CAMERA] Capture error: {e}")
                time.sleep(0.1)
    
    def capture_frame(self):
        """Get the latest captured frame (non-blocking)"""
        if not self.is_running:
            return None
        
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None
    
    def capture_latest_frame(self):
        """Get the latest captured frame - same as capture_frame with threaded capture"""
        return self.capture_frame()
    
    def stop(self):
        """Stop capture thread and release the camera"""
        self.is_running = False
        
        # Wait for capture thread to finish
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=1.0)
        
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
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if not os.path.exists(cascade_path):
                raise FileNotFoundError(f"Haar cascade file not found at: {cascade_path}")
            
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            
            if self.face_cascade.empty():
                raise RuntimeError("Failed to load Haar cascade classifier - file may be corrupted")
                
        except Exception as e:
            raise RuntimeError(
                f"Could not load face detection model: {e}\n"
                f"Please ensure OpenCV is properly installed with: pip install opencv-python"
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
    
    def train_single_person(self, person_name, progress_callback=None):
        """Train only a single person's images and add to existing model (incremental training)"""
        from imutils import paths
        
        person_folder = os.path.join(self.dataset_path, person_name)
        if not os.path.exists(person_folder):
            return False, f"No folder found for {person_name}"
        
        image_paths = list(paths.list_images(person_folder))
        if not image_paths:
            return False, f"No images found for {person_name}"
        
        new_encodings = []
        new_names = []
        
        for i, image_path in enumerate(image_paths):
            if progress_callback:
                progress_callback(i + 1, len(image_paths), image_path)
            
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            boxes = face_recognition.face_locations(rgb, model="hog")
            encodings = face_recognition.face_encodings(rgb, boxes)
            
            for encoding in encodings:
                new_encodings.append(encoding)
                new_names.append(person_name)
        
        if not new_encodings:
            return False, f"No faces detected in images for {person_name}"
        
        # Remove any existing encodings for this person (in case of re-training)
        indices_to_keep = [i for i, name in enumerate(self.known_names) if name != person_name]
        self.known_encodings = [self.known_encodings[i] for i in indices_to_keep]
        self.known_names = [self.known_names[i] for i in indices_to_keep]
        
        # Add new encodings
        self.known_encodings.extend(new_encodings)
        self.known_names.extend(new_names)
        
        # Save updated model
        data = {"encodings": self.known_encodings, "names": self.known_names}
        with open(self.encodings_path, "wb") as f:
            f.write(pickle.dumps(data))
        
        # Clear cache since we have new encodings
        self.face_cache.clear()
        
        return True, f"Added {len(new_encodings)} encodings for {person_name}"
    
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
    
    def estimate_head_pose(self, frame, face_location):
        """
        Estimate head pose (yaw, pitch) using facial landmarks.
        Returns (yaw, pitch) in degrees, or None if landmarks not detected.
        Yaw: negative = looking left, positive = looking right
        Pitch: negative = looking down, positive = looking up
        """
        top, right, bottom, left = face_location
        
        # Get facial landmarks for this face
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = [(int(top), int(right), int(bottom), int(left))]
        
        try:
            landmarks_list = face_recognition.face_landmarks(rgb_frame, face_locations)
            if not landmarks_list:
                return None
            
            landmarks = landmarks_list[0]
            
            # Get key points for pose estimation
            nose_bridge = landmarks.get('nose_bridge', [])
            nose_tip = landmarks.get('nose_tip', [])
            left_eye = landmarks.get('left_eye', [])
            right_eye = landmarks.get('right_eye', [])
            
            if not all([nose_bridge, nose_tip, left_eye, right_eye]):
                return None
            
            # Calculate center points
            left_eye_center = np.mean(left_eye, axis=0)
            right_eye_center = np.mean(right_eye, axis=0)
            nose_bridge_top = np.array(nose_bridge[0])
            nose_tip_center = np.mean(nose_tip, axis=0)
            
            # Eye center
            eye_center = (left_eye_center + right_eye_center) / 2
            
            # Estimate yaw (left-right rotation)
            # Compare nose tip horizontal position relative to eye center
            face_width = right_eye_center[0] - left_eye_center[0]
            if face_width > 0:
                nose_offset = (nose_tip_center[0] - eye_center[0]) / face_width
                yaw = nose_offset * 60  # Scale to approximate degrees
            else:
                yaw = 0
            
            # Estimate pitch (up-down rotation)
            # Compare nose tip vertical position relative to nose bridge
            face_height = bottom - top
            if face_height > 0:
                # Vertical distance from nose bridge to nose tip
                nose_length = nose_tip_center[1] - nose_bridge_top[1]
                expected_nose_length = face_height * 0.25
                if expected_nose_length > 0:
                    pitch_ratio = (nose_length - expected_nose_length) / expected_nose_length
                    pitch = -pitch_ratio * 45  # Scale to approximate degrees
                else:
                    pitch = 0
            else:
                pitch = 0
            
            return (float(yaw), float(pitch))
            
        except Exception as e:
            return None
    
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
        
        # Auto-capture Face ID style registration
        self.auto_capture_mode = False
        self.auto_capture_target = 100  # Target number of photos
        self.auto_capture_interval = 0.15  # Seconds between captures
        self.last_auto_capture = 0
        self.captured_poses = []  # List of (yaw, pitch) for captured photos
        self.current_pose = None  # Current detected head pose
        self.pose_zones_captured = {}  # Track which pose zones have been captured
        self.pose_guidance_text = ""
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        self.faces_detected = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # User management state
        self.person_map = {}  # Maps listbox index to person name
        
        # Memory management - periodic cache cleanup
        self.last_cache_cleanup = time.time()
        self.cache_cleanup_interval = 60  # seconds
        
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
        """Create the main kiosk interface - commercial Apple-like design"""
        # Configure root window
        self.root.configure(bg=Config.COLOR_BG)
        
        # Main container - edge to edge
        self.main_frame = tk.Frame(self.root, bg=Config.COLOR_BG)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # ===== TOP BAR - Minimal status bar =====
        top_bar = tk.Frame(self.main_frame, bg=Config.COLOR_BG, height=60)
        top_bar.pack(fill=tk.X, padx=30, pady=(20, 0))
        top_bar.pack_propagate(False)
        
        # Time display - left side, elegant
        self.time_label = tk.Label(
            top_bar,
            text="",
            font=(Config.FONT_FAMILY, 16),
            fg=Config.COLOR_TEXT_SECONDARY,
            bg=Config.COLOR_BG
        )
        self.time_label.pack(side=tk.LEFT, pady=15)
        self.update_time()
        
        # Live indicator - right side
        self.fps_indicator = tk.Label(
            top_bar,
            text="● LIVE",
            font=(Config.FONT_FAMILY, 11, "bold"),
            fg=Config.COLOR_GRANTED,
            bg=Config.COLOR_BG
        )
        self.fps_indicator.pack(side=tk.RIGHT, pady=15)
        
        # ===== CENTER CONTENT - Camera and Status =====
        center_frame = tk.Frame(self.main_frame, bg=Config.COLOR_BG)
        center_frame.pack(fill=tk.BOTH, expand=True, padx=30, pady=20)
        
        # Camera feed - large, centered, rounded appearance
        camera_wrapper = tk.Frame(center_frame, bg=Config.COLOR_BG)
        camera_wrapper.pack(expand=True)
        
        # Video container with card styling
        self.video_container = tk.Frame(
            camera_wrapper, 
            bg=Config.COLOR_CARD,
            highlightbackground=Config.COLOR_BORDER,
            highlightthickness=1
        )
        self.video_container.pack(padx=2, pady=2)
        self.video_container.pack_propagate(False)
        
        # Calculate video size based on screen
        screen_height = self.root.winfo_screenheight()
        video_height = int(screen_height * 0.55)
        video_width = int(video_height * 4 / 3)
        self.video_container.config(width=video_width, height=video_height)
        
        self.video_label = tk.Label(self.video_container, bg="#000000")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # ===== STATUS OVERLAY - Floating status badge =====
        # This sits below the camera
        status_frame = tk.Frame(camera_wrapper, bg=Config.COLOR_BG)
        status_frame.pack(pady=(25, 0))
        
        # Status card - pill shaped appearance
        self.status_card = tk.Frame(
            status_frame, 
            bg=Config.COLOR_CARD,
            highlightbackground=Config.COLOR_BORDER,
            highlightthickness=1
        )
        self.status_card.pack()
        
        status_inner = tk.Frame(self.status_card, bg=Config.COLOR_CARD)
        status_inner.pack(padx=40, pady=20)
        self.status_frame = status_inner
        
        # Horizontal status layout
        self.status_icon_label = tk.Label(
            status_inner,
            text="◉",
            font=(Config.FONT_FAMILY, 36),
            fg=Config.COLOR_SCANNING,
            bg=Config.COLOR_CARD
        )
        self.status_icon_label.pack(side=tk.LEFT, padx=(0, 15))
        
        status_text_frame = tk.Frame(status_inner, bg=Config.COLOR_CARD)
        status_text_frame.pack(side=tk.LEFT)
        
        self.status_text_label = tk.Label(
            status_text_frame,
            text="Ready to Scan",
            font=(Config.FONT_FAMILY, 20, "bold"),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_CARD,
            anchor="w"
        )
        self.status_text_label.pack(anchor="w")
        
        self.status_detail_label = tk.Label(
            status_text_frame,
            text="Look at the camera",
            font=(Config.FONT_FAMILY, 13),
            fg=Config.COLOR_TEXT_SECONDARY,
            bg=Config.COLOR_CARD,
            anchor="w"
        )
        self.status_detail_label.pack(anchor="w")
        
        # Store text frame reference
        self.status_text_frame = status_text_frame
        
        # ===== BOTTOM BAR =====
        bottom_bar = tk.Frame(self.main_frame, bg=Config.COLOR_BG, height=80)
        bottom_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=30, pady=(0, 25))
        bottom_bar.pack_propagate(False)
        
        # Left side - Settings button (subtle)
        self.admin_btn = tk.Button(
            bottom_bar,
            text="⚙",
            font=(Config.FONT_FAMILY, 20),
            fg=Config.COLOR_TEXT_TERTIARY,
            bg=Config.COLOR_BG,
            activebackground=Config.COLOR_BG,
            activeforeground=Config.COLOR_TEXT_SECONDARY,
            bd=0,
            cursor="hand2",
            command=self.show_admin_login
        )
        self.admin_btn.pack(side=tk.LEFT, pady=20)
        
        # Center - Company/Building name
        self.title_label = tk.Label(
            bottom_bar,
            text="SECURE ENTRY",
            font=(Config.FONT_FAMILY, 12, "bold"),
            fg=Config.COLOR_TEXT_TERTIARY,
            bg=Config.COLOR_BG,
            anchor="center"
        )
        self.title_label.pack(side=tk.LEFT, expand=True, pady=20)
        
        # Right side - User count
        self.info_label = tk.Label(
            bottom_bar,
            text=f"{len(self.face_system.get_trained_persons())} users",
            font=(Config.FONT_FAMILY, 11),
            fg=Config.COLOR_TEXT_TERTIARY,
            bg=Config.COLOR_BG
        )
        self.info_label.pack(side=tk.RIGHT, pady=20)
        
        # Hidden activity log for this view (shown in admin panel)
        self.log_listbox = tk.Listbox(self.main_frame)
        self.log_listbox.pack_forget()
        
        # Load recent log entries
        self.update_log_display()
    
    def update_time(self):
        """Update the time display"""
        current_time = datetime.now().strftime("%H:%M")
        current_date = datetime.now().strftime("%a, %b %d")
        self.time_label.config(text=f"{current_time}  ·  {current_date}")
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
        
        # Helper to update all widget backgrounds
        def update_bg(bg_color, icon_fg, text_fg, detail_fg):
            self.status_card.config(bg=bg_color, highlightbackground=bg_color)
            self.status_frame.config(bg=bg_color)
            self.status_text_frame.config(bg=bg_color)
            self.status_icon_label.config(bg=bg_color, fg=icon_fg)
            self.status_text_label.config(bg=bg_color, fg=text_fg)
            self.status_detail_label.config(bg=bg_color, fg=detail_fg)
        
        if status == "granted":
            self.status_icon_label.config(text="✓")
            self.status_text_label.config(text=f"Welcome, {name}")
            self.status_detail_label.config(text="Access granted")
            update_bg(Config.COLOR_GRANTED, "#FFFFFF", "#FFFFFF", "#E8F5E9")
            # Flash effect - briefly highlight then return
            self.root.after(2500, lambda: self.set_status("scanning"))
            
        elif status == "denied":
            self.status_icon_label.config(text="✕")
            self.status_text_label.config(text="Not Recognized")
            self.status_detail_label.config(text="Access denied")
            update_bg(Config.COLOR_DENIED, "#FFFFFF", "#FFFFFF", "#FFEBEE")
            self.root.after(2500, lambda: self.set_status("scanning"))
            
        else:  # scanning
            self.status_icon_label.config(text="◉")
            self.status_text_label.config(text="Ready to Scan")
            self.status_detail_label.config(text="Look at the camera")
            update_bg(Config.COLOR_CARD, Config.COLOR_SCANNING, Config.COLOR_TEXT, Config.COLOR_TEXT_SECONDARY)
            self.status_card.config(highlightbackground=Config.COLOR_BORDER)
    
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
                
                # Use capture_latest_frame to get the newest frame and reduce latency
                frame = self.camera.capture_latest_frame()
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
                    # Registration mode - detect face and estimate pose
                    faces = self.face_system.detect_faces_fast(frame)
                    
                    frame_height, frame_width = display_frame.shape[:2]
                    
                    if len(faces) == 1:
                        top, right, bottom, left = faces[0]
                        
                        # Estimate head pose
                        pose = self.face_system.estimate_head_pose(frame, faces[0])
                        self.current_pose = pose
                        
                        if pose:
                            yaw, pitch = pose
                            
                            # Determine pose zone (8 zones like Face ID)
                            zone = self.get_pose_zone(yaw, pitch)
                            
                            # Auto-capture logic
                            if self.auto_capture_mode:
                                now = time.time()
                                if now - self.last_auto_capture >= self.auto_capture_interval:
                                    if self.captured_count < self.auto_capture_target:
                                        # Check if this pose angle is sufficiently different
                                        if self.should_capture_pose(yaw, pitch):
                                            filepath = self.face_system.save_face_image(frame, self.registration_name)
                                            self.captured_count += 1
                                            self.captured_poses.append((yaw, pitch))
                                            self.pose_zones_captured[zone] = self.pose_zones_captured.get(zone, 0) + 1
                                            self.last_auto_capture = now
                                            self.root.after(0, self.update_registration_ui)
                                    else:
                                        # Auto-capture complete
                                        self.root.after(0, self.complete_auto_registration)
                            
                            # Draw pose indicator on frame
                            self.draw_pose_indicator(display_frame, yaw, pitch, zone)
                            
                            # Draw face box with pose-based color
                            color = (0, 255, 0) if zone in self.pose_zones_captured else (0, 255, 255)
                            cv2.rectangle(display_frame, (int(left), int(top)), (int(right), int(bottom)), color, 3)
                            
                            # Guidance text
                            guidance = self.get_pose_guidance()
                            cv2.putText(display_frame, guidance, (frame_width // 2 - 150, frame_height - 40),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        else:
                            cv2.rectangle(display_frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 255), 3)
                    
                    elif len(faces) == 0:
                        cv2.putText(display_frame, "No face detected - position your face", (frame_width // 2 - 180, frame_height // 2),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                    else:
                        cv2.putText(display_frame, "Multiple faces detected - only one person", (frame_width // 2 - 200, frame_height // 2),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Show registration info overlay
                    self.draw_registration_overlay(display_frame)
                
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
                
                # Periodic cache cleanup to prevent memory buildup
                now = time.time()
                if now - self.last_cache_cleanup > self.cache_cleanup_interval:
                    expired_count = self.face_system.face_cache.cleanup_expired()
                    if expired_count and expired_count > 0:
                        print(f"Cache cleanup: removed {expired_count} expired entries")
                    self.last_cache_cleanup = now
                
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
        if USE_PICAMERA:
            frame_rgb = frame
        else:
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
        """Show admin login dialog with custom dialog for fullscreen compatibility"""
        # Create custom dialog (simpledialog has issues with fullscreen on macOS)
        login_dialog = tk.Toplevel(self.root)
        login_dialog.title("Admin Login")
        login_dialog.geometry("350x180")
        login_dialog.configure(bg=Config.COLOR_BG)
        login_dialog.resizable(False, False)
        
        # Center on screen
        login_dialog.update_idletasks()
        x = (login_dialog.winfo_screenwidth() - 350) // 2
        y = (login_dialog.winfo_screenheight() - 180) // 2
        login_dialog.geometry(f"350x180+{x}+{y}")
        
        # Make modal
        login_dialog.transient(self.root)
        login_dialog.grab_set()
        login_dialog.focus_set()
        
        # Password entry result
        result = {'password': None}
        
        # Content frame
        content = tk.Frame(login_dialog, bg=Config.COLOR_BG)
        content.pack(fill=tk.BOTH, expand=True, padx=25, pady=20)
        
        tk.Label(
            content,
            text="Enter admin password:",
            font=(Config.FONT_FAMILY, 14),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_BG
        ).pack(anchor=tk.W, pady=(0, 10))
        
        password_entry = tk.Entry(
            content,
            font=(Config.FONT_FAMILY, 14),
            show='*',
            relief=tk.FLAT,
            bg=Config.COLOR_CARD,
            fg=Config.COLOR_TEXT,
            insertbackground=Config.COLOR_TEXT
        )
        password_entry.pack(fill=tk.X, ipady=8)
        password_entry.focus_set()
        
        # Button frame
        btn_frame = tk.Frame(content, bg=Config.COLOR_BG)
        btn_frame.pack(fill=tk.X, pady=(20, 0))
        
        def on_ok(event=None):
            result['password'] = password_entry.get()
            login_dialog.destroy()
        
        def on_cancel(event=None):
            login_dialog.destroy()
        
        cancel_btn = tk.Button(
            btn_frame,
            text="Cancel",
            font=(Config.FONT_FAMILY, 12),
            fg=Config.COLOR_TEXT_SECONDARY,
            bg=Config.COLOR_CARD,
            activeforeground=Config.COLOR_TEXT,
            activebackground=Config.COLOR_CARD,
            relief=tk.FLAT,
            cursor="hand2",
            width=10,
            command=on_cancel
        )
        cancel_btn.pack(side=tk.LEFT)
        
        ok_btn = tk.Button(
            btn_frame,
            text="OK",
            font=(Config.FONT_FAMILY, 12),
            fg="white",
            bg=Config.COLOR_SCANNING,
            activeforeground="white",
            activebackground=Config.COLOR_SCANNING,
            relief=tk.FLAT,
            cursor="hand2",
            width=10,
            command=on_ok
        )
        ok_btn.pack(side=tk.RIGHT)
        
        # Bind Enter and Escape keys
        password_entry.bind('<Return>', on_ok)
        login_dialog.bind('<Escape>', on_cancel)
        
        # Wait for dialog to close
        self.root.wait_window(login_dialog)
        
        # Check password
        password = result['password']
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
        
        # Auto-capture button (Face ID style)
        self.auto_capture_btn = tk.Button(
            btn_frame,
            text="⟳ Auto Capture (100 photos)",
            font=(Config.FONT_FAMILY, 13),
            fg="#FFFFFF",
            bg="#5856D6",  # Purple like Face ID
            activebackground="#4744c4",
            activeforeground="#FFFFFF",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.start_auto_capture,
            state=tk.DISABLED
        )
        self.auto_capture_btn.pack(fill=tk.X, pady=3, ipady=8)
        
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
        
        # Auto-train option
        self.auto_train_var = tk.BooleanVar(value=True)
        auto_train_frame = tk.Frame(inner, bg=Config.COLOR_CARD)
        auto_train_frame.pack(fill=tk.X, pady=(15, 0))
        
        self.auto_train_check = tk.Checkbutton(
            auto_train_frame,
            text="Auto-train after capture (recommended)",
            variable=self.auto_train_var,
            font=(Config.FONT_FAMILY, 11),
            fg=Config.COLOR_TEXT_SECONDARY,
            bg=Config.COLOR_CARD,
            activebackground=Config.COLOR_CARD,
            selectcolor=Config.COLOR_CARD
        )
        self.auto_train_check.pack(anchor=tk.W)
        
        # Tips
        tips_frame = tk.Frame(parent, bg=Config.COLOR_BG)
        tips_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(
            tips_frame,
            text="Face ID Style Registration:\n• Use Auto Capture for best results\n• Slowly rotate your head in a circle\n• Cover all angles: up, down, left, right",
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
        
        # Filter card
        filter_card = tk.Frame(parent, bg=Config.COLOR_CARD, highlightbackground=Config.COLOR_BORDER, highlightthickness=1)
        filter_card.pack(fill=tk.X, padx=20, pady=(0, 10))
        
        filter_inner = tk.Frame(filter_card, bg=Config.COLOR_CARD)
        filter_inner.pack(fill=tk.X, padx=15, pady=15)
        
        # Date filter row
        date_row = tk.Frame(filter_inner, bg=Config.COLOR_CARD)
        date_row.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(
            date_row,
            text="Date Range:",
            font=(Config.FONT_FAMILY, 11),
            fg=Config.COLOR_TEXT_SECONDARY,
            bg=Config.COLOR_CARD
        ).pack(side=tk.LEFT)
        
        # From date
        tk.Label(
            date_row,
            text="From:",
            font=(Config.FONT_FAMILY, 11),
            fg=Config.COLOR_TEXT_SECONDARY,
            bg=Config.COLOR_CARD
        ).pack(side=tk.LEFT, padx=(15, 5))
        
        self.log_date_from = tk.Entry(
            date_row,
            font=(Config.FONT_FAMILY, 11),
            width=12,
            bg=Config.COLOR_CARD_SECONDARY,
            fg=Config.COLOR_TEXT,
            relief=tk.FLAT,
            highlightbackground=Config.COLOR_BORDER,
            highlightthickness=1
        )
        self.log_date_from.pack(side=tk.LEFT, ipady=4)
        self.log_date_from.insert(0, "YYYY-MM-DD")
        self.log_date_from.bind("<FocusIn>", lambda e: self._clear_placeholder(e, "YYYY-MM-DD"))
        self.log_date_from.bind("<FocusOut>", lambda e: self._restore_placeholder(e, "YYYY-MM-DD"))
        
        # To date
        tk.Label(
            date_row,
            text="To:",
            font=(Config.FONT_FAMILY, 11),
            fg=Config.COLOR_TEXT_SECONDARY,
            bg=Config.COLOR_CARD
        ).pack(side=tk.LEFT, padx=(15, 5))
        
        self.log_date_to = tk.Entry(
            date_row,
            font=(Config.FONT_FAMILY, 11),
            width=12,
            bg=Config.COLOR_CARD_SECONDARY,
            fg=Config.COLOR_TEXT,
            relief=tk.FLAT,
            highlightbackground=Config.COLOR_BORDER,
            highlightthickness=1
        )
        self.log_date_to.pack(side=tk.LEFT, ipady=4)
        self.log_date_to.insert(0, "YYYY-MM-DD")
        self.log_date_to.bind("<FocusIn>", lambda e: self._clear_placeholder(e, "YYYY-MM-DD"))
        self.log_date_to.bind("<FocusOut>", lambda e: self._restore_placeholder(e, "YYYY-MM-DD"))
        
        # Name filter row
        name_row = tk.Frame(filter_inner, bg=Config.COLOR_CARD)
        name_row.pack(fill=tk.X, pady=(0, 10))
        
        tk.Label(
            name_row,
            text="Filter by Name:",
            font=(Config.FONT_FAMILY, 11),
            fg=Config.COLOR_TEXT_SECONDARY,
            bg=Config.COLOR_CARD
        ).pack(side=tk.LEFT)
        
        # Name dropdown with autocomplete
        self.log_name_var = tk.StringVar(value="All")
        unique_names = ["All"] + self.access_log.get_unique_names()
        
        self.log_name_combo = ttk.Combobox(
            name_row,
            textvariable=self.log_name_var,
            values=unique_names,
            font=(Config.FONT_FAMILY, 11),
            width=20,
            state="readonly"
        )
        self.log_name_combo.pack(side=tk.LEFT, padx=(10, 0))
        
        # Filter buttons
        btn_row = tk.Frame(filter_inner, bg=Config.COLOR_CARD)
        btn_row.pack(fill=tk.X)
        
        tk.Button(
            btn_row,
            text="Apply Filter",
            font=(Config.FONT_FAMILY, 11),
            fg="white",
            bg=Config.COLOR_SCANNING,
            activeforeground="white",
            activebackground="#0056b3",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.apply_log_filter
        ).pack(side=tk.LEFT, ipadx=10, ipady=4)
        
        tk.Button(
            btn_row,
            text="Clear Filter",
            font=(Config.FONT_FAMILY, 11),
            fg=Config.COLOR_TEXT_SECONDARY,
            bg=Config.COLOR_CARD_SECONDARY,
            activeforeground=Config.COLOR_TEXT,
            activebackground=Config.COLOR_CARD_SECONDARY,
            relief=tk.FLAT,
            cursor="hand2",
            command=self.clear_log_filter
        ).pack(side=tk.LEFT, padx=(10, 0), ipadx=10, ipady=4)
        
        # Quick date buttons
        tk.Button(
            btn_row,
            text="Today",
            font=(Config.FONT_FAMILY, 10),
            fg=Config.COLOR_SCANNING,
            bg=Config.COLOR_CARD,
            activeforeground=Config.COLOR_SCANNING,
            bd=0,
            cursor="hand2",
            command=lambda: self.set_log_date_range(0)
        ).pack(side=tk.RIGHT, padx=5)
        
        tk.Button(
            btn_row,
            text="7 Days",
            font=(Config.FONT_FAMILY, 10),
            fg=Config.COLOR_SCANNING,
            bg=Config.COLOR_CARD,
            activeforeground=Config.COLOR_SCANNING,
            bd=0,
            cursor="hand2",
            command=lambda: self.set_log_date_range(7)
        ).pack(side=tk.RIGHT, padx=5)
        
        tk.Button(
            btn_row,
            text="30 Days",
            font=(Config.FONT_FAMILY, 10),
            fg=Config.COLOR_SCANNING,
            bg=Config.COLOR_CARD,
            activeforeground=Config.COLOR_SCANNING,
            bd=0,
            cursor="hand2",
            command=lambda: self.set_log_date_range(30)
        ).pack(side=tk.RIGHT, padx=5)
        
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
        
        # Populate with all entries initially
        self.populate_log_listbox(self.access_log.get_recent(100))
    
    def _clear_placeholder(self, event, placeholder):
        """Clear placeholder text on focus"""
        if event.widget.get() == placeholder:
            event.widget.delete(0, tk.END)
            event.widget.config(fg=Config.COLOR_TEXT)
    
    def _restore_placeholder(self, event, placeholder):
        """Restore placeholder text if empty"""
        if not event.widget.get():
            event.widget.insert(0, placeholder)
            event.widget.config(fg=Config.COLOR_TEXT_SECONDARY)
    
    def set_log_date_range(self, days_back):
        """Set date range for quick filters"""
        from datetime import timedelta
        today = datetime.now().date()
        
        self.log_date_to.delete(0, tk.END)
        self.log_date_to.insert(0, today.isoformat())
        self.log_date_to.config(fg=Config.COLOR_TEXT)
        
        if days_back == 0:
            # Today only
            self.log_date_from.delete(0, tk.END)
            self.log_date_from.insert(0, today.isoformat())
        else:
            from_date = today - timedelta(days=days_back)
            self.log_date_from.delete(0, tk.END)
            self.log_date_from.insert(0, from_date.isoformat())
        
        self.log_date_from.config(fg=Config.COLOR_TEXT)
        self.apply_log_filter()
    
    def apply_log_filter(self):
        """Apply filters and refresh log display"""
        from datetime import date
        
        # Parse date from
        date_from = None
        date_from_str = self.log_date_from.get()
        if date_from_str and date_from_str != "YYYY-MM-DD":
            try:
                date_from = date.fromisoformat(date_from_str)
            except ValueError:
                messagebox.showerror("Invalid Date", "From date must be in YYYY-MM-DD format")
                return
        
        # Parse date to
        date_to = None
        date_to_str = self.log_date_to.get()
        if date_to_str and date_to_str != "YYYY-MM-DD":
            try:
                date_to = date.fromisoformat(date_to_str)
            except ValueError:
                messagebox.showerror("Invalid Date", "To date must be in YYYY-MM-DD format")
                return
        
        # Get name filter
        name_filter = None
        selected_name = self.log_name_var.get()
        if selected_name and selected_name != "All":
            name_filter = selected_name
        
        # Get filtered entries
        entries = self.access_log.get_filtered(date_from, date_to, name_filter, count=100)
        self.populate_log_listbox(entries)
    
    def clear_log_filter(self):
        """Clear all filters and show all entries"""
        self.log_date_from.delete(0, tk.END)
        self.log_date_from.insert(0, "YYYY-MM-DD")
        self.log_date_from.config(fg=Config.COLOR_TEXT_SECONDARY)
        
        self.log_date_to.delete(0, tk.END)
        self.log_date_to.insert(0, "YYYY-MM-DD")
        self.log_date_to.config(fg=Config.COLOR_TEXT_SECONDARY)
        
        self.log_name_var.set("All")
        
        self.populate_log_listbox(self.access_log.get_recent(100))
    
    def populate_log_listbox(self, entries):
        """Populate the log listbox with entries"""
        self.admin_log_listbox.delete(0, tk.END)
        for entry in entries:
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
        self.person_map = {}  # Reset the mapping
        
        persons = self.face_system.get_registered_persons()
        for idx, (name, count) in enumerate(persons):
            self.person_map[idx] = name  # Store index-to-name mapping
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
        self.auto_capture_mode = False
        self.captured_poses = []
        self.pose_zones_captured = {}
        
        self.start_reg_btn.config(state=tk.DISABLED)
        self.capture_btn.config(state=tk.NORMAL)
        self.auto_capture_btn.config(state=tk.NORMAL)
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
        """Stop face registration mode and optionally auto-train"""
        person_name = self.registration_name
        captured = self.captured_count
        
        self.registration_mode = False
        self.registration_name = ""
        self.auto_capture_mode = False
        self.captured_poses = []
        self.pose_zones_captured = {}
        
        self.start_reg_btn.config(state=tk.NORMAL)
        self.capture_btn.config(state=tk.DISABLED)
        self.auto_capture_btn.config(state=tk.DISABLED)
        self.stop_reg_btn.config(state=tk.DISABLED)
        self.reg_name_entry.config(state=tk.NORMAL)
        self.reg_name_entry.delete(0, tk.END)
        
        self.refresh_manage_list()
        
        # Auto-train the new person if option is enabled and photos were captured
        if hasattr(self, 'auto_train_var') and self.auto_train_var.get() and captured > 0 and person_name:
            self.train_single_person(person_name)
    
    # ==================== AUTO-CAPTURE FACE ID STYLE ====================
    
    def start_auto_capture(self):
        """Start automatic Face ID style capture"""
        if not self.registration_mode:
            return
        
        self.auto_capture_mode = True
        self.last_auto_capture = time.time()
        self.captured_poses = []
        self.pose_zones_captured = {}
        
        # Update UI
        self.capture_btn.config(state=tk.DISABLED)
        self.auto_capture_btn.config(text="⏹ Stop Auto Capture", bg=Config.COLOR_DENIED, command=self.stop_auto_capture)
        self.reg_count_label.config(text="Move your head slowly in a circle...")
    
    def stop_auto_capture(self):
        """Stop automatic capture"""
        self.auto_capture_mode = False
        self.capture_btn.config(state=tk.NORMAL)
        self.auto_capture_btn.config(text="⟳ Auto Capture (100 photos)", bg="#5856D6", command=self.start_auto_capture)
        self.update_registration_ui()
    
    def get_pose_zone(self, yaw, pitch):
        """
        Determine which of 9 pose zones the current head pose falls into.
        Returns zone name: 'center', 'up', 'down', 'left', 'right', 
                          'up_left', 'up_right', 'down_left', 'down_right'
        """
        yaw_threshold = 12
        pitch_threshold = 10
        
        if abs(yaw) < yaw_threshold and abs(pitch) < pitch_threshold:
            return 'center'
        elif pitch > pitch_threshold:
            if yaw < -yaw_threshold:
                return 'up_left'
            elif yaw > yaw_threshold:
                return 'up_right'
            else:
                return 'up'
        elif pitch < -pitch_threshold:
            if yaw < -yaw_threshold:
                return 'down_left'
            elif yaw > yaw_threshold:
                return 'down_right'
            else:
                return 'down'
        else:
            if yaw < -yaw_threshold:
                return 'left'
            else:
                return 'right'
    
    def should_capture_pose(self, yaw, pitch):
        """
        Determine if we should capture at this pose.
        Ensures variety in captured angles.
        """
        min_angle_diff = 5  # Minimum angle difference from previous captures
        
        # Always capture first few photos
        if len(self.captured_poses) < 10:
            return True
        
        # Check if this pose is different enough from recent captures
        for prev_yaw, prev_pitch in self.captured_poses[-15:]:
            if abs(yaw - prev_yaw) < min_angle_diff and abs(pitch - prev_pitch) < min_angle_diff:
                return False
        
        return True
    
    def get_pose_guidance(self):
        """Get guidance text based on which zones need more coverage"""
        zone_targets = {
            'center': 15, 'up': 12, 'down': 12, 'left': 12, 'right': 12,
            'up_left': 10, 'up_right': 10, 'down_left': 10, 'down_right': 10
        }
        
        # Find zones that need more photos
        needed_zones = []
        for zone, target in zone_targets.items():
            captured = self.pose_zones_captured.get(zone, 0)
            if captured < target:
                needed_zones.append(zone)
        
        if not needed_zones:
            return "Great coverage! Keep moving for more variety"
        
        # Prioritize guidance
        zone_guidance = {
            'center': "Look straight at camera",
            'up': "Look up slightly",
            'down': "Look down slightly", 
            'left': "Turn head left",
            'right': "Turn head right",
            'up_left': "Look up and left",
            'up_right': "Look up and right",
            'down_left': "Look down and left",
            'down_right': "Look down and right"
        }
        
        # Return guidance for first needed zone
        return zone_guidance.get(needed_zones[0], "Move your head slowly")
    
    def draw_pose_indicator(self, frame, yaw, pitch, zone):
        """Draw a visual indicator showing current head pose and progress"""
        frame_height, frame_width = frame.shape[:2]
        
        # Draw pose compass in top-right corner
        compass_size = 80
        compass_x = frame_width - compass_size - 20
        compass_y = 60
        
        # Draw compass background circle
        cv2.circle(frame, (compass_x, compass_y), compass_size // 2, (50, 50, 50), -1)
        cv2.circle(frame, (compass_x, compass_y), compass_size // 2, (100, 100, 100), 2)
        
        # Draw zone indicators (9 zones)
        zones_pos = {
            'up_left': (-20, -20), 'up': (0, -25), 'up_right': (20, -20),
            'left': (-25, 0), 'center': (0, 0), 'right': (25, 0),
            'down_left': (-20, 20), 'down': (0, 25), 'down_right': (20, 20)
        }
        
        for z, (dx, dy) in zones_pos.items():
            color = (0, 255, 0) if z in self.pose_zones_captured else (80, 80, 80)
            pos = (compass_x + dx, compass_y + dy)
            radius = 8 if z == 'center' else 5
            cv2.circle(frame, pos, radius, color, -1)
        
        # Draw current pose position (red dot)
        pose_x = int(compass_x + (yaw / 30) * 30)  # Scale yaw to compass
        pose_y = int(compass_y - (pitch / 20) * 25)  # Scale pitch to compass
        cv2.circle(frame, (pose_x, pose_y), 6, (0, 0, 255), -1)
        cv2.circle(frame, (pose_x, pose_y), 6, (255, 255, 255), 2)
        
        # Draw progress bar
        progress = min(1.0, self.captured_count / self.auto_capture_target)
        bar_width = 200
        bar_height = 8
        bar_x = frame_width // 2 - bar_width // 2
        bar_y = 25
        
        # Background
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        # Progress
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + int(bar_width * progress), bar_y + bar_height), (0, 255, 0), -1)
        
        # Progress text
        progress_text = f"{self.captured_count}/{self.auto_capture_target}"
        cv2.putText(frame, progress_text, (bar_x + bar_width + 10, bar_y + 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def draw_registration_overlay(self, frame):
        """Draw registration mode overlay"""
        frame_height, frame_width = frame.shape[:2]
        
        # Mode indicator
        mode_text = "AUTO CAPTURE" if self.auto_capture_mode else "REGISTRATION"
        mode_color = (200, 100, 200) if self.auto_capture_mode else (0, 255, 255)
        cv2.putText(frame, mode_text, (15, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, mode_color, 2)
        
        # Person name
        cv2.putText(frame, f"Name: {self.registration_name}", (15, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    def update_registration_ui(self):
        """Update the registration UI with current progress"""
        zones_filled = len(self.pose_zones_captured)
        if self.auto_capture_mode:
            self.reg_count_label.config(
                text=f"{self.captured_count}/{self.auto_capture_target} photos • {zones_filled}/9 angles"
            )
        else:
            self.reg_count_label.config(text=f"{self.captured_count} photos captured")
    
    def complete_auto_registration(self):
        """Called when auto-capture reaches target"""
        self.auto_capture_mode = False
        self.capture_btn.config(state=tk.NORMAL)
        self.auto_capture_btn.config(text="✓ Complete!", bg=Config.COLOR_GRANTED, state=tk.DISABLED)
        
        zones_filled = len(self.pose_zones_captured)
        self.reg_count_label.config(
            text=f"✓ {self.captured_count} photos from {zones_filled} angles"
        )
        
        messagebox.showinfo(
            "Auto Capture Complete", 
            f"Successfully captured {self.captured_count} photos covering {zones_filled} head pose angles.\\n\\n"
            "Click 'Stop' to finish and train the model."
        )
    
    # ==================== END AUTO-CAPTURE ====================
    
    def train_single_person(self, person_name):
        """Train only a single person (incremental training)"""
        self.reg_count_label.config(text=f"Training {person_name}...")
        
        def training_thread():
            success, message = self.face_system.train_single_person(person_name)
            self.root.after(0, lambda: self.single_training_complete(success, message))
        
        thread = threading.Thread(target=training_thread, daemon=True)
        thread.start()
    
    def single_training_complete(self, success, message):
        """Handle single person training completion"""
        if success:
            self.reg_count_label.config(text=f"✓ {message}")
            messagebox.showinfo("Training Complete", message)
        else:
            self.reg_count_label.config(text=f"✗ Training failed")
            messagebox.showerror("Training Failed", message)
        
        # Reset after a delay
        self.root.after(3000, lambda: self.reg_count_label.config(text="0 photos captured"))
    
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
        
        selected_index = selection[0]
        
        # Use person_map for robust name lookup (avoids string parsing issues)
        name = self.person_map.get(selected_index)
        if name is None:
            messagebox.showerror("Error", "Could not identify selected person. Please refresh and try again.")
            return
        
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
