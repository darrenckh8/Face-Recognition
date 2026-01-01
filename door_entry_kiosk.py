import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import cv2
import os
import threading
import time
from datetime import datetime
import pickle
import numpy as np
from PIL import Image, ImageTk
import json
import queue
from concurrent.futures import ThreadPoolExecutor
import gc

# Try to import insightface - required for this application
try:
    from insightface.app import FaceAnalysis
    import insightface
except ImportError:
    print("Error: insightface library not found. Please install it with: pip install insightface onnxruntime")
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
    FULLSCREEN = False
    WINDOW_TITLE = "Door Entry System"
    
    # Admin Settings
    ADMIN_PASSWORD = "admin123"  # Change this in production!
    
    # Camera Settings
    CAMERA_RESOLUTION = (640, 480)
    
    # Recognition Settings
    RECOGNITION_THRESHOLD = 0.45  # Cosine similarity threshold for InsightFace (0.4-0.5 recommended)
    COOLDOWN_SECONDS = 5  # Prevent repeated access logs for same person
    
    # Performance Settings
    RECOGNITION_INTERVAL_FRAMES = 3  # Only run recognition every N frames
    FACE_CACHE_TTL = 5  # Seconds to cache a recognized face
    FACE_CACHE_MAX_SIZE = 50  # Maximum cache entries before LRU eviction
    FACE_POSITION_TOLERANCE = 80  # Pixels tolerance for face position matching
    DETECTION_SCALE_FACTOR = 2  # Scale down factor for faster processing
    USE_FAST_DETECTION = False  # Disabled - InsightFace handles detection efficiently
    
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
    """Caches recognized faces with LRU eviction and memory limits.
    
    Memory optimizations:
    - Max size limit with LRU eviction
    - Stores only person name reference, not full 512D embedding
    - Periodic cleanup of expired entries
    """
    
    def __init__(self, ttl=None, position_tolerance=None, max_size=None):
        self.ttl = ttl or Config.FACE_CACHE_TTL
        self.position_tolerance = position_tolerance or Config.FACE_POSITION_TOLERANCE
        self.max_size = max_size or Config.FACE_CACHE_MAX_SIZE
        self.cache = {}  # {cache_key: {name, confidence, location, timestamp, last_access}}
        self.lock = threading.Lock()
        self._access_order = []  # Track access order for LRU eviction
    
    def _get_position_key(self, location):
        """Generate a grid-based position key for face location"""
        top, right, bottom, left = location
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2
        # Round to grid cells based on tolerance
        grid_x = center_x // self.position_tolerance
        grid_y = center_y // self.position_tolerance
        return (grid_x, grid_y)
    
    def _evict_lru(self):
        """Evict least recently used entries if cache exceeds max size"""
        while len(self.cache) >= self.max_size and self._access_order:
            oldest_key = self._access_order.pop(0)
            if oldest_key in self.cache:
                del self.cache[oldest_key]
    
    def _update_access_order(self, key):
        """Update LRU access order for a key"""
        if key in self._access_order:
            self._access_order.remove(key)
        self._access_order.append(key)
    
    def _find_nearby_cache(self, location):
        """Find a cached face near the given location"""
        top, right, bottom, left = location
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2
        
        now = time.time()
        best_match = None
        best_distance = float('inf')
        best_key = None
        
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
                    best_key = key
            
            # Remove expired entries
            for key in expired_keys:
                del self.cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
            
            # Update access order for found entry
            if best_key is not None:
                self._update_access_order(best_key)
        
        return best_match
    
    def get(self, location):
        """Get cached recognition result for a face at given location"""
        return self._find_nearby_cache(location)
    
    def put(self, location, name, confidence, encoding=None):
        """Cache a recognition result (encoding parameter kept for API compatibility but not stored)"""
        key = self._get_position_key(location)
        with self.lock:
            # Evict if at capacity
            if key not in self.cache:
                self._evict_lru()
            
            self.cache[key] = {
                'name': name,
                'confidence': confidence,
                'location': location,
                'timestamp': time.time()
                # Note: encoding is NOT stored to save memory
            }
            self._update_access_order(key)
    
    def clear(self):
        """Clear all cached entries"""
        with self.lock:
            self.cache.clear()
            self._access_order.clear()
    
    def cleanup_expired(self):
        """Remove expired entries from cache, returns count of removed entries"""
        now = time.time()
        with self.lock:
            expired_keys = [k for k, v in self.cache.items() if now - v['timestamp'] > self.ttl]
            for key in expired_keys:
                del self.cache[key]
                if key in self._access_order:
                    self._access_order.remove(key)
            return len(expired_keys)
    
    def size(self):
        """Return current cache size"""
        return len(self.cache)


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
        
        # Thread-safe frame storage with double buffering
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.capture_thread = None
        
        # Pre-allocated frame buffers for zero-copy operation
        self._frame_buffer_a = None
        self._frame_buffer_b = None
        self._active_buffer = 'a'  # Which buffer is currently being written to
        self._frame_id = 0  # Increments on each new frame
    
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
        """Background thread that continuously captures frames with buffer reuse"""
        # Pre-allocate buffers on first frame
        buffers_initialized = False
        
        while self.is_running:
            try:
                if self.use_picamera:
                    frame = self.camera.capture_array()
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                else:
                    ret, frame = self.camera.read()
                    if not ret:
                        continue
                
                # Initialize buffers on first valid frame
                if not buffers_initialized:
                    h, w = frame.shape[:2]
                    self._frame_buffer_a = np.empty((h, w, 3), dtype=np.uint8)
                    self._frame_buffer_b = np.empty((h, w, 3), dtype=np.uint8)
                    buffers_initialized = True
                
                # Copy to inactive buffer (swap on next iteration)
                with self.frame_lock:
                    if self._active_buffer == 'a':
                        np.copyto(self._frame_buffer_b, frame)
                        self.current_frame = self._frame_buffer_b
                        self._active_buffer = 'b'
                    else:
                        np.copyto(self._frame_buffer_a, frame)
                        self.current_frame = self._frame_buffer_a
                        self._active_buffer = 'a'
                    self._frame_id += 1
                    
            except Exception as e:
                print(f"[CAMERA] Capture error: {e}")
                time.sleep(0.1)
    
    def capture_frame(self, copy=True):
        """Get the latest captured frame (non-blocking)
        
        Args:
            copy: If True, returns a copy (safe for modification).
                  If False, returns direct reference (faster, read-only use).
        
        Returns:
            Frame array or None if no frame available
        """
        if not self.is_running:
            return None
        
        with self.frame_lock:
            if self.current_frame is not None:
                if copy:
                    return self.current_frame.copy()
                else:
                    # Return direct reference - caller must not modify
                    return self.current_frame
        return None
    
    def get_frame_id(self):
        """Get current frame ID for change detection"""
        with self.frame_lock:
            return self._frame_id
    
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
    """Core face recognition logic with performance optimizations using InsightFace"""
    
    def __init__(self, dataset_path=None, encodings_path=None):
        self.dataset_path = dataset_path or Config.DATASET_PATH
        self.encodings_path = encodings_path or Config.ENCODINGS_PATH
        # Also support compressed numpy format
        self.encodings_path_npz = self.encodings_path.replace('.pickle', '.npz')
        
        self.known_encodings = []
        self.known_names = []
        self.known_encodings_normalized = None  # Pre-normalized matrix for fast comparison
        self.cv_scaler = Config.DETECTION_SCALE_FACTOR
        
        # Pre-allocated similarity array for memory efficiency
        self._similarity_buffer = None
        
        # Initialize InsightFace model (buffalo_s is lighter for Raspberry Pi)
        try:
            self.face_app = FaceAnalysis(name='buffalo_s', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            print("[INFO] InsightFace model (buffalo_s) loaded successfully")
        except Exception as e:
            raise RuntimeError(
                f"Could not load InsightFace model: {e}\n"
                f"Please ensure insightface is properly installed with: pip install insightface onnxruntime"
            )
        
        # Keep Haar cascade for fast detection fallback (optional)
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
            else:
                self.face_cascade = None
        except Exception:
            self.face_cascade = None
        
        # Performance: Face cache to avoid repeated recognition
        self.face_cache = FaceCache()
        
        # Performance: Frame counter for skipping
        self.frame_count = 0
        
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
        
        self.load_encodings()
    
    def load_encodings(self):
        """Load face encodings with support for compressed numpy format (faster) or pickle fallback"""
        # Try compressed numpy format first (3-5x faster loading)
        if os.path.exists(self.encodings_path_npz):
            try:
                data = np.load(self.encodings_path_npz, allow_pickle=True)
                self.known_encodings = list(data['encodings'])
                self.known_names = list(data['names'])
                self._update_normalized_encodings()
                print(f"[INFO] Loaded {len(self.known_encodings)} encodings from compressed format")
                return True
            except Exception as e:
                print(f"Error loading compressed encodings: {e}")
        
        # Fallback to pickle format
        if os.path.exists(self.encodings_path):
            try:
                with open(self.encodings_path, "rb") as f:
                    data = pickle.loads(f.read())
                # Convert lists back to numpy arrays for InsightFace compatibility
                self.known_encodings = [np.array(enc) for enc in data["encodings"]]
                self.known_names = data["names"]
                # Pre-normalize encodings for vectorized cosine similarity
                self._update_normalized_encodings()
                
                # Convert to compressed format for faster future loads
                self._save_compressed_encodings()
                return True
            except Exception as e:
                print(f"Error loading encodings: {e}")
                return False
        return False
    
    def _save_compressed_encodings(self):
        """Save encodings in compressed numpy format for faster loading"""
        if self.known_encodings:
            try:
                np.savez_compressed(
                    self.encodings_path_npz,
                    encodings=np.array(self.known_encodings),
                    names=np.array(self.known_names)
                )
                print(f"[INFO] Saved compressed encodings to {self.encodings_path_npz}")
            except Exception as e:
                print(f"Error saving compressed encodings: {e}")
    
    def _update_normalized_encodings(self):
        """Pre-normalize all known encodings for fast vectorized comparison"""
        if len(self.known_encodings) > 0:
            encodings_matrix = np.array(self.known_encodings)
            norms = np.linalg.norm(encodings_matrix, axis=1, keepdims=True)
            self.known_encodings_normalized = encodings_matrix / norms
            # Pre-allocate similarity buffer
            self._similarity_buffer = np.empty(len(self.known_encodings), dtype=np.float32)
        else:
            self.known_encodings_normalized = None
            self._similarity_buffer = None
    
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
            
            # InsightFace expects BGR images
            faces = self.face_app.get(image)
            
            for face in faces:
                if face.embedding is not None:
                    known_encodings.append(face.embedding)
                    known_names.append(name)
        
        if not known_encodings:
            return False, "No faces detected in any images"
        
        # Save in both formats: pickle for backward compatibility, compressed for speed
        data = {"encodings": [enc.tolist() for enc in known_encodings], "names": known_names}
        with open(self.encodings_path, "wb") as f:
            f.write(pickle.dumps(data))
        
        self.known_encodings = [np.array(enc) for enc in known_encodings]
        self.known_names = known_names
        
        # Pre-normalize for fast comparison
        self._update_normalized_encodings()
        
        # Also save compressed format for faster loading
        self._save_compressed_encodings()
        
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
            
            # InsightFace expects BGR images
            faces = self.face_app.get(image)
            
            for face in faces:
                if face.embedding is not None:
                    new_encodings.append(face.embedding)
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
        
        # Save updated model (convert numpy arrays to lists for serialization)
        data = {"encodings": [enc.tolist() if hasattr(enc, 'tolist') else enc for enc in self.known_encodings], "names": self.known_names}
        with open(self.encodings_path, "wb") as f:
            f.write(pickle.dumps(data))
        
        # Pre-normalize for fast comparison
        self._update_normalized_encodings()
        
        # Also save compressed format for faster loading
        self._save_compressed_encodings()
        
        # Clear cache since we have new encodings
        self.face_cache.clear()
        
        return True, f"Added {len(new_encodings)} encodings for {person_name}"
    
    def remove_person_from_model(self, person_name):
        """Remove a person's encodings from the trained model"""
        if not self.known_names:
            return False, "No trained model exists"
        
        # Check if person exists in model
        if person_name not in self.known_names:
            return True, f"{person_name} not found in model (already removed or never trained)"
        
        # Count encodings to remove
        count_before = len(self.known_encodings)
        
        # Remove all encodings for this person
        indices_to_keep = [i for i, name in enumerate(self.known_names) if name != person_name]
        self.known_encodings = [self.known_encodings[i] for i in indices_to_keep]
        self.known_names = [self.known_names[i] for i in indices_to_keep]
        
        count_removed = count_before - len(self.known_encodings)
        
        # Save updated model
        if self.known_encodings:
            data = {"encodings": [enc.tolist() if hasattr(enc, 'tolist') else enc for enc in self.known_encodings], "names": self.known_names}
            with open(self.encodings_path, "wb") as f:
                f.write(pickle.dumps(data))
            # Pre-normalize for fast comparison
            self._update_normalized_encodings()
            # Also save compressed format
            self._save_compressed_encodings()
        else:
            # No encodings left, remove both files
            if os.path.exists(self.encodings_path):
                os.remove(self.encodings_path)
            if os.path.exists(self.encodings_path_npz):
                os.remove(self.encodings_path_npz)
            self.known_encodings_normalized = None
            self._similarity_buffer = None
        
        # Clear cache
        self.face_cache.clear()
        
        return True, f"Removed {count_removed} encodings for {person_name}"
    
    def detect_faces_fast(self, frame):
        """Fast face detection using Haar cascade (no recognition)"""
        # Fall back to InsightFace if Haar cascade not available
        if self.face_cascade is None or self.face_cascade.empty():
            return self.detect_faces_robust(frame)
        
        small_frame = cv2.resize(frame, (0, 0), fx=1/self.cv_scaler, fy=1/self.cv_scaler)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with Haar cascade (fast)
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Convert to (top, right, bottom, left) format and scale up
        locations = []
        for (x, y, w, h) in faces:
            # Scale back to original frame size
            top = y * self.cv_scaler
            right = (x + w) * self.cv_scaler
            bottom = (y + h) * self.cv_scaler
            left = x * self.cv_scaler
            locations.append((top, right, bottom, left))
        
        return locations
    
    def detect_faces_robust(self, frame):
        """
        Robust face detection using InsightFace.
        Better at detecting faces at various angles - ideal for registration.
        More reliable than Haar cascade.
        """
        # InsightFace expects BGR images (which is what OpenCV provides)
        faces = self.face_app.get(frame)
        
        # Convert InsightFace bbox format to (top, right, bottom, left)
        face_locations = []
        for face in faces:
            bbox = face.bbox.astype(int)
            left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
            face_locations.append((top, right, bottom, left))
        
        return face_locations
    
    def detect_faces_combined(self, frame):
        """
        Combined detection: try fast Haar first, fall back to robust dlib.
        Best of both worlds for registration mode.
        """
        # Try fast detection first
        faces = self.detect_faces_fast(frame)
        
        # If no faces found, try robust detection
        if not faces:
            faces = self.detect_faces_robust(frame)
        
        return faces
    
    def recognize_faces(self, frame, force_recognition=False):
        """Detect and recognize faces in a frame using InsightFace"""
        self.frame_count += 1
        
        if not self.known_encodings or self.known_encodings_normalized is None:
            return frame, []
        
        results = []
        
        # Skip frames for performance (unless forced)
        if not force_recognition and self.frame_count % Config.RECOGNITION_INTERVAL_FRAMES != 0:
            return frame, []
        
        # Use InsightFace for detection and embedding extraction (BGR input - OpenCV default)
        faces = self.face_app.get(frame)
        
        # Process each detected face
        for face in faces:
            if face.embedding is None:
                continue
            
            face_encoding = face.embedding
            bbox = face.bbox.astype(int)
            left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
            location = (top, right, bottom, left)
            
            # Check cache first
            cached = self.face_cache.get(location)
            if cached and not force_recognition:
                results.append({
                    'name': cached['name'],
                    'confidence': cached['confidence'],
                    'location': location,
                    'from_cache': True
                })
                continue
            
            # Vectorized cosine similarity with pre-normalized encodings
            name = "Unknown"
            confidence = 0.0
            
            # Normalize current face encoding
            face_norm = face_encoding / np.linalg.norm(face_encoding)
            
            # Use pre-allocated buffer for similarity computation (avoids allocation per face)
            if self._similarity_buffer is not None and len(self._similarity_buffer) == len(self.known_encodings):
                np.dot(self.known_encodings_normalized, face_norm, out=self._similarity_buffer)
                similarities = self._similarity_buffer
            else:
                # Fallback if buffer size mismatch
                similarities = np.dot(self.known_encodings_normalized, face_norm)
            
            best_match_index = np.argmax(similarities)
            best_similarity = similarities[best_match_index]
            
            # Use configured threshold
            if best_similarity > Config.RECOGNITION_THRESHOLD:
                name = self.known_names[best_match_index]
                confidence = float(best_similarity)
            
            # Cache this result (encoding not stored to save memory)
            self.face_cache.put(location, name, confidence)
            
            results.append({
                'name': name,
                'confidence': confidence,
                'location': location,
                'from_cache': False
            })
        
        return frame, results
    
    def clear_cache(self):
        """Clear the face recognition cache"""
        self.face_cache.clear()


# ==================== RECOGNITION WORKER ====================
class RecognitionWorker:
    """
    Handles face recognition in a separate thread pool to prevent blocking the camera feed.
    Uses queue-based architecture for passing frames and receiving results.
    """
    
    def __init__(self, face_system, max_workers=2):
        self.face_system = face_system
        self.max_workers = max_workers
        
        # Queues for communication
        self.frame_queue = queue.Queue(maxsize=2)  # Limit queue size to prevent memory buildup
        self.result_queue = queue.Queue(maxsize=10)
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="RecognitionWorker")
        
        # Control flags
        self.is_running = False
        self.worker_thread = None
        
        # Track pending futures
        self.pending_futures = []
        self.futures_lock = threading.Lock()
        
        # Performance metrics
        self.frames_processed = 0
        self.frames_dropped = 0
        self.avg_processing_time = 0.0
    
    def start(self):
        """Start the recognition worker"""
        self.is_running = True
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        print("[WORKER] Recognition worker started with {} threads".format(self.max_workers))
    
    def stop(self):
        """Stop the recognition worker and cleanup"""
        self.is_running = False
        
        # Clear queues
        self._clear_queue(self.frame_queue)
        self._clear_queue(self.result_queue)
        
        # Shutdown executor
        self.executor.shutdown(wait=False, cancel_futures=True)
        
        # Wait for worker thread
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=1.0)
        
        print("[WORKER] Recognition worker stopped")
    
    def _clear_queue(self, q):
        """Clear all items from a queue"""
        try:
            while True:
                q.get_nowait()
        except queue.Empty:
            pass
    
    def submit_frame(self, frame, frame_id=None, force_recognition=False):
        """
        Submit a frame for recognition processing.
        Non-blocking - drops frame if queue is full.
        
        Args:
            frame: BGR image frame from camera
            frame_id: Optional frame identifier for tracking
            force_recognition: If True, bypasses frame skipping
            
        Returns:
            True if frame was accepted, False if dropped
        """
        if not self.is_running:
            return False
        
        try:
            # Non-blocking put - drop frame if queue is full
            self.frame_queue.put_nowait({
                'frame': frame.copy(),
                'frame_id': frame_id or time.time(),
                'force_recognition': force_recognition,
                'timestamp': time.time()
            })
            return True
        except queue.Full:
            self.frames_dropped += 1
            return False
    
    def get_results(self, timeout=0):
        """
        Get recognition results from the result queue.
        Non-blocking by default.
        
        Args:
            timeout: Maximum time to wait (0 for non-blocking)
            
        Returns:
            List of result dicts, or empty list if none available
        """
        results = []
        try:
            while True:
                if timeout > 0:
                    result = self.result_queue.get(timeout=timeout)
                    timeout = 0  # Only wait on first get
                else:
                    result = self.result_queue.get_nowait()
                results.append(result)
        except queue.Empty:
            pass
        return results
    
    def _worker_loop(self):
        """Main worker loop that dispatches frames to thread pool"""
        while self.is_running:
            try:
                # Get frame from queue (blocking with timeout)
                try:
                    frame_data = self.frame_queue.get(timeout=0.1)
                except queue.Empty:
                    # Clean up completed futures
                    self._cleanup_futures()
                    continue
                
                # Check if this is detection-only or full recognition
                if frame_data.get('detection_only', False):
                    # Detection only (for registration mode)
                    future = self.executor.submit(
                        self._process_detection,
                        frame_data['frame'],
                        frame_data['frame_id'],
                        frame_data.get('detection_method', 'combined'),
                        frame_data['timestamp']
                    )
                else:
                    # Full recognition
                    future = self.executor.submit(
                        self._process_frame,
                        frame_data['frame'],
                        frame_data['frame_id'],
                        frame_data.get('force_recognition', False),
                        frame_data['timestamp']
                    )
                
                with self.futures_lock:
                    self.pending_futures.append(future)
                
                # Cleanup completed futures periodically
                self._cleanup_futures()
                
            except Exception as e:
                print(f"[WORKER] Error in worker loop: {e}")
    
    def _process_frame(self, frame, frame_id, force_recognition, submit_time):
        """Process a single frame (runs in thread pool)"""
        try:
            start_time = time.time()
            
            # Perform recognition
            _, results = self.face_system.recognize_faces(frame, force_recognition=force_recognition)
            
            processing_time = time.time() - start_time
            latency = time.time() - submit_time
            
            # Update metrics
            self.frames_processed += 1
            self.avg_processing_time = (self.avg_processing_time * 0.9) + (processing_time * 0.1)
            
            # Put results in output queue
            result_data = {
                'frame_id': frame_id,
                'results': results,
                'processing_time': processing_time,
                'latency': latency,
                'timestamp': time.time()
            }
            
            try:
                self.result_queue.put_nowait(result_data)
            except queue.Full:
                # Result queue full, discard oldest and add new
                try:
                    self.result_queue.get_nowait()
                    self.result_queue.put_nowait(result_data)
                except queue.Empty:
                    pass
            
            return result_data
            
        except Exception as e:
            print(f"[WORKER] Error processing frame: {e}")
            return None
    
    def _cleanup_futures(self):
        """Remove completed futures from tracking list"""
        with self.futures_lock:
            self.pending_futures = [f for f in self.pending_futures if not f.done()]
    
    def get_stats(self):
        """Get worker statistics"""
        return {
            'frames_processed': self.frames_processed,
            'frames_dropped': self.frames_dropped,
            'avg_processing_time': self.avg_processing_time,
            'pending_tasks': len(self.pending_futures),
            'frame_queue_size': self.frame_queue.qsize(),
            'result_queue_size': self.result_queue.qsize()
        }
    
    # ===== Async Face Detection Methods =====
    
    def submit_detection(self, frame, frame_id=None, detection_method='combined'):
        """
        Submit a frame for face detection only (no recognition).
        Used during registration mode.
        
        Args:
            frame: BGR image frame from camera
            frame_id: Optional frame identifier for tracking
            detection_method: 'fast', 'robust', or 'combined'
            
        Returns:
            True if frame was accepted, False if dropped
        """
        if not self.is_running:
            return False
        
        try:
            self.frame_queue.put_nowait({
                'frame': frame.copy(),
                'frame_id': frame_id or time.time(),
                'detection_only': True,
                'detection_method': detection_method,
                'timestamp': time.time()
            })
            return True
        except queue.Full:
            self.frames_dropped += 1
            return False
    
    def _process_detection(self, frame, frame_id, detection_method, submit_time):
        """Process a single frame for detection only (runs in thread pool)"""
        try:
            start_time = time.time()
            
            # Perform detection based on method
            if detection_method == 'fast':
                faces = self.face_system.detect_faces_fast(frame)
            elif detection_method == 'robust':
                faces = self.face_system.detect_faces_robust(frame)
            else:  # combined
                faces = self.face_system.detect_faces_combined(frame)
            
            processing_time = time.time() - start_time
            latency = time.time() - submit_time
            
            # Update metrics
            self.frames_processed += 1
            self.avg_processing_time = (self.avg_processing_time * 0.9) + (processing_time * 0.1)
            
            # Put results in output queue
            result_data = {
                'frame_id': frame_id,
                'faces': faces,
                'detection_only': True,
                'processing_time': processing_time,
                'latency': latency,
                'timestamp': time.time()
            }
            
            try:
                self.result_queue.put_nowait(result_data)
            except queue.Full:
                try:
                    self.result_queue.get_nowait()
                    self.result_queue.put_nowait(result_data)
                except queue.Empty:
                    pass
            
            return result_data
            
        except Exception as e:
            print(f"[WORKER] Error processing detection: {e}")
            return None

# ==================== ON-SCREEN KEYBOARD ====================
class OnScreenKeyboard:
    """Touch-friendly on-screen keyboard embedded in parent container"""
    
    _active_keyboard = None  # Track active keyboard to prevent duplicates
    
    def __init__(self, container, entry_widget, root_window):
        # Close any existing keyboard first
        if OnScreenKeyboard._active_keyboard is not None:
            try:
                OnScreenKeyboard._active_keyboard.close()
            except:
                pass
        
        self.container = container
        self.entry = entry_widget
        self.root_window = root_window
        self.shift_on = False
        self.keyboard_frame = None
        self.all_buttons = []  # Track all keyboard buttons
        
        self._create_keyboard()
        
        # Bind events for auto-close
        self.entry.bind('<Return>', lambda e: self.close())
        
        OnScreenKeyboard._active_keyboard = self
    
    def _is_keyboard_widget(self, widget):
        """Check if widget is part of keyboard"""
        if widget == self.keyboard_frame or widget == self.entry:
            return True
        if widget in self.all_buttons:
            return True
        try:
            parent = widget.master
            while parent:
                if parent == self.keyboard_frame:
                    return True
                parent = parent.master
        except:
            pass
        return False
    
    def _create_keyboard(self):
        """Create the keyboard layout embedded in container"""
        # Create keyboard frame at bottom of container
        self.keyboard_frame = tk.Frame(self.container, bg=Config.COLOR_BORDER)
        self.keyboard_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        main_frame = tk.Frame(self.keyboard_frame, bg=Config.COLOR_BG)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Keyboard rows - all keys same size except space bar
        rows = [
            ['1', '2', '3', '4', '5', '6', '7', '8', '9', '0'],
            ['q', 'w', 'e', 'r', 't', 'y', 'u', 'i', 'o', 'p'],
            ['a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', '@'],
            ['⇧', 'z', 'x', 'c', 'v', 'b', 'n', 'm', '-', '⌫'],
            ['✕', ' ', '.', '✓']
        ]
        
        for row in rows:
            row_frame = tk.Frame(main_frame, bg=Config.COLOR_BG)
            row_frame.pack(fill=tk.X, pady=1)
            
            for key in row:
                bg_color = Config.COLOR_CARD
                fg_color = Config.COLOR_TEXT
                if key == '✓':
                    bg_color = Config.COLOR_GRANTED
                    fg_color = "#FFFFFF"
                elif key == '✕':
                    bg_color = Config.COLOR_DENIED
                    fg_color = "#FFFFFF"
                
                # Use expand for equal sizing, space bar gets more expand weight
                expand_weight = 3 if key == ' ' else 1
                
                btn = tk.Button(
                    row_frame,
                    text=key if key != ' ' else '␣',
                    font=(Config.FONT_FAMILY, 13),
                    bg=bg_color,
                    fg=fg_color,
                    activebackground=Config.COLOR_CARD_SECONDARY,
                    relief=tk.FLAT,
                    highlightthickness=0,
                    command=lambda k=key: self._on_key_press(k)
                )
                btn.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=1, ipady=8)
                self.all_buttons.append(btn)
                
                if key == '⇧':
                    self.shift_btn = btn
    
    def _on_key_press(self, key):
        """Handle key press"""
        if key == '✓':
            self.close()
        elif key == '✕':
            self.entry.delete(0, tk.END)
            self.close()
        elif key == '⌫':
            current = self.entry.get()
            self.entry.delete(0, tk.END)
            self.entry.insert(0, current[:-1])
        elif key == '⇧':
            self.shift_on = not self.shift_on
            self.shift_btn.config(bg=Config.COLOR_SCANNING if self.shift_on else Config.COLOR_CARD,
                                  fg="#FFFFFF" if self.shift_on else Config.COLOR_TEXT)
        elif key == ' ':
            self.entry.insert(tk.END, ' ')
        else:
            char = key.upper() if self.shift_on else key
            self.entry.insert(tk.END, char)
            if self.shift_on:
                self.shift_on = False
                self.shift_btn.config(bg=Config.COLOR_CARD, fg=Config.COLOR_TEXT)
        
        # Keep focus on entry
        self.entry.focus_set()
    
    def close(self):
        """Close the keyboard"""
        try:
            self.entry.unbind('<Return>')
        except:
            pass
        OnScreenKeyboard._active_keyboard = None
        self.all_buttons = []
        if self.keyboard_frame:
            try:
                self.keyboard_frame.destroy()
            except:
                pass
        self.keyboard_frame = None


def show_keyboard(container, entry_widget, root_window=None):
    """Show the on-screen keyboard embedded in container"""
    if root_window is None:
        root_window = container
    OnScreenKeyboard(container, entry_widget, root_window)


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
            self.root.geometry("480x800")
            self.root.resizable(False, False)
        
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
        self.auto_capture_interval = 0.2  # Seconds between captures
        self.last_auto_capture = 0
        
        # Face ID style zone tracking (based on face position, not pose estimation)
        self.zone_captures = {'center': 0, 'left': 0, 'right': 0, 'up': 0, 'down': 0}
        self.current_zone = 'center'
        self.zone_targets = {'center': 30, 'left': 18, 'right': 18, 'up': 17, 'down': 17}  # = 100 total
        
        # Async detection results cache for registration mode
        self._cached_detection_faces = []
        self._last_detection_frame_id = -1
        
        # Training state - prevents concurrent InsightFace access
        self.is_training = False
        self.reg_process_locked = False  # Lock during capture, training, and encoding
        
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
        
        # Initialize recognition worker for non-blocking face recognition
        self.recognition_worker = RecognitionWorker(self.face_system, max_workers=2)
        
        # Latest recognition results for display (updated from worker)
        self.latest_recognition_results = []
        self.recognition_results_lock = threading.Lock()
        
        # Frame processing optimization - pre-allocated buffers
        self._display_buffer = np.empty((330, 440, 3), dtype=np.uint8)  # Pre-allocated display buffer
        self._display_buffer_rgb = np.empty((330, 440, 3), dtype=np.uint8)  # Pre-allocated RGB buffer
        self._last_displayed_frame_id = -1  # Track which frame was last displayed
        self._cached_imgtk = None  # Cached PhotoImage to avoid recreation
        
        # Create GUI
        self.create_kiosk_interface()
        
        # Start recognition worker
        self.recognition_worker.start()
        
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
        top_bar = tk.Frame(self.main_frame, bg=Config.COLOR_BG, height=40)
        top_bar.pack(fill=tk.X, padx=15, pady=(10, 0))
        top_bar.pack_propagate(False)
        
        # Time display - left side, elegant
        self.time_label = tk.Label(
            top_bar,
            text="",
            font=(Config.FONT_FAMILY, 12),
            fg=Config.COLOR_TEXT_SECONDARY,
            bg=Config.COLOR_BG
        )
        self.time_label.pack(side=tk.LEFT, pady=8)
        self.update_time()
        
        # Live indicator - right side
        self.fps_indicator = tk.Label(
            top_bar,
            text="● LIVE",
            font=(Config.FONT_FAMILY, 9, "bold"),
            fg=Config.COLOR_GRANTED,
            bg=Config.COLOR_BG
        )
        self.fps_indicator.pack(side=tk.RIGHT, pady=8)
        
        # ===== CENTER CONTENT - Camera and Status =====
        center_frame = tk.Frame(self.main_frame, bg=Config.COLOR_BG)
        center_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
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
        self.video_container.pack()
        self.video_container.pack_propagate(False)
        
        # Fixed video size for consistent UI (scaled for 480px window)
        video_width = 440
        video_height = 330
        self.video_container.config(width=video_width, height=video_height)
        
        self.video_label = tk.Label(self.video_container, bg="#000000")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # ===== STATUS OVERLAY - Floating status badge =====
        # This sits below the camera
        status_frame = tk.Frame(camera_wrapper, bg=Config.COLOR_BG)
        status_frame.pack(pady=(15, 0))
        
        # Status card - pill shaped appearance
        self.status_card = tk.Frame(
            status_frame, 
            bg=Config.COLOR_CARD,
            highlightbackground=Config.COLOR_BORDER,
            highlightthickness=1
        )
        self.status_card.pack()
        
        status_inner = tk.Frame(self.status_card, bg=Config.COLOR_CARD)
        status_inner.pack(padx=20, pady=12)
        self.status_frame = status_inner
        
        # Horizontal status layout
        self.status_icon_label = tk.Label(
            status_inner,
            text="◉",
            font=(Config.FONT_FAMILY, 24),
            fg=Config.COLOR_SCANNING,
            bg=Config.COLOR_CARD
        )
        self.status_icon_label.pack(side=tk.LEFT, padx=(0, 10))
        
        status_text_frame = tk.Frame(status_inner, bg=Config.COLOR_CARD)
        status_text_frame.pack(side=tk.LEFT)
        
        self.status_text_label = tk.Label(
            status_text_frame,
            text="Ready to Scan",
            font=(Config.FONT_FAMILY, 14, "bold"),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_CARD,
            anchor="w"
        )
        self.status_text_label.pack(anchor="w")
        
        self.status_detail_label = tk.Label(
            status_text_frame,
            text="Look at the camera",
            font=(Config.FONT_FAMILY, 10),
            fg=Config.COLOR_TEXT_SECONDARY,
            bg=Config.COLOR_CARD,
            anchor="w"
        )
        self.status_detail_label.pack(anchor="w")
        
        # Store text frame reference
        self.status_text_frame = status_text_frame
        
        # ===== BOTTOM BAR =====
        bottom_bar = tk.Frame(self.main_frame, bg=Config.COLOR_BG, height=50)
        bottom_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=15, pady=(0, 10))
        bottom_bar.pack_propagate(False)
        
        # Left side - Settings button (subtle)
        self.admin_btn = tk.Button(
            bottom_bar,
            text="⚙",
            font=(Config.FONT_FAMILY, 16),
            fg=Config.COLOR_TEXT_TERTIARY,
            bg=Config.COLOR_BG,
            activebackground=Config.COLOR_BG,
            activeforeground=Config.COLOR_TEXT_SECONDARY,
            bd=0,
            cursor="hand2",
            command=self.show_admin_login
        )
        self.admin_btn.pack(side=tk.LEFT, pady=10)
        
        # Center - Company/Building name
        self.title_label = tk.Label(
            bottom_bar,
            text="EDUWEL",
            font=(Config.FONT_FAMILY, 10, "bold"),
            fg=Config.COLOR_TEXT_TERTIARY,
            bg=Config.COLOR_BG,
            anchor="center"
        )
        self.title_label.pack(side=tk.LEFT, expand=True, pady=10)
        
        # Right side - User count
        self.info_label = tk.Label(
            bottom_bar,
            text=f"{len(self.face_system.get_trained_persons())} Users",
            font=(Config.FONT_FAMILY, 9),
            fg=Config.COLOR_TEXT_TERTIARY,
            bg=Config.COLOR_BG
        )
        self.info_label.pack(side=tk.RIGHT, pady=10)
        
        # Hidden activity log for this view (shown in admin panel)
        self.log_listbox = tk.Listbox(self.main_frame)
        self.log_listbox.pack_forget()
        
        # Load recent log entries
        self.update_log_display()
    
    def update_time(self):
        """Update the time display"""
        current_time = datetime.now().strftime("%H:%M:%S")
        current_date = datetime.now().strftime("%a, %b %d %Y")
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
            
        elif status == "active_scanning":
            self.status_icon_label.config(text="◎")
            self.status_text_label.config(text="Scanning...")
            self.status_detail_label.config(text="Please hold still")
            update_bg(Config.COLOR_CARD, Config.COLOR_SCANNING, Config.COLOR_TEXT, Config.COLOR_TEXT_SECONDARY)
            self.status_card.config(highlightbackground=Config.COLOR_SCANNING)
            
        else:  # scanning (ready state)
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
        """Main camera loop running in a separate thread - non-blocking recognition"""
        try:
            self.camera.start()
            frame_time = time.time()
            frame_counter = 0
            
            # Pre-allocated buffer for display frame modifications
            display_frame = None
            
            while self.is_running:
                loop_start = time.time()
                
                # Get the newest frame from the camera (zero-copy for read-only use)
                frame = self.camera.capture_frame(copy=False)
                if frame is None:
                    continue
                
                # Only copy when we need to modify (draw overlays)
                # Reuse display_frame buffer if same size
                if display_frame is None or display_frame.shape != frame.shape:
                    display_frame = np.empty_like(frame)
                np.copyto(display_frame, frame)
                
                frame_counter += 1
                
                if self.is_scanning and not self.registration_mode and not self.admin_mode:
                    # Skip recognition if training is in progress (prevents segfault)
                    if self.is_training:
                        cv2.putText(display_frame, "Training in progress...", (50, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                    else:
                        # Non-blocking: Submit frame to recognition worker every N frames
                        if frame_counter % Config.RECOGNITION_INTERVAL_FRAMES == 0:
                            self.recognition_worker.submit_frame(frame, frame_id=frame_counter)
                        
                        # Non-blocking: Poll for recognition results
                        worker_results = self.recognition_worker.get_results()
                        
                        # Process any available results
                        for result_data in worker_results:
                            results = result_data.get('results', [])
                            
                            # Track performance metrics
                            self.faces_detected = len(results)
                            cache_hits_this_frame = sum(1 for r in results if r.get('from_cache', False))
                            cache_misses_this_frame = len(results) - cache_hits_this_frame
                            self.cache_hits += cache_hits_this_frame
                            self.cache_misses += cache_misses_this_frame
                            
                            # Store latest results for display
                            with self.recognition_results_lock:
                                self.latest_recognition_results = results
                            
                            # Update status based on whether faces are detected
                            if len(results) > 0 and self.current_status not in ("granted", "denied"):
                                self.root.after(0, lambda: self.set_status("active_scanning"))
                            elif len(results) == 0 and self.current_status == "active_scanning":
                                self.root.after(0, lambda: self.set_status("scanning"))
                            
                            for result in results:
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
                                else:
                                    # Check if we should log denied access (only for non-cached results)
                                    if name == "Unknown" and self.current_status in ("scanning", "active_scanning") and not from_cache:
                                        now = time.time()
                                        if "Unknown" not in self.last_access or (now - self.last_access["Unknown"]) > Config.COOLDOWN_SECONDS:
                                            self.last_access["Unknown"] = now
                                            self.root.after(0, self.deny_access)
                
                elif self.registration_mode:
                    # Registration mode - async face detection for non-blocking operation
                    frame_height, frame_width = display_frame.shape[:2]
                    
                    # Submit frame for async detection every few frames
                    if frame_counter % 2 == 0:  # Every 2nd frame
                        self.recognition_worker.submit_detection(frame, frame_id=frame_counter)
                    
                    # Poll for detection results (non-blocking)
                    detection_results = self.recognition_worker.get_results()
                    for result_data in detection_results:
                        if result_data.get('detection_only', False):
                            self._cached_detection_faces = result_data.get('faces', [])
                            self._last_detection_frame_id = result_data.get('frame_id', -1)
                    
                    # Use cached detection results
                    faces = self._cached_detection_faces
                    
                    if self.auto_capture_mode:
                        # Active capture mode - show Face ID overlay
                        frame_center_x = frame_width // 2
                        frame_center_y = frame_height // 2
                        
                        if len(faces) == 1:
                            top, right, bottom, left = faces[0]
                            
                            # Calculate face center
                            face_center_x = (left + right) / 2
                            face_center_y = (top + bottom) / 2
                            
                            # Determine zone based on face position in frame
                            offset_x = (face_center_x - frame_center_x) / (frame_width / 2)
                            offset_y = (face_center_y - frame_center_y) / (frame_height / 2)
                            
                            # Determine zone with generous thresholds
                            zone_threshold = 0.15
                            if abs(offset_x) < zone_threshold and abs(offset_y) < zone_threshold:
                                self.current_zone = 'center'
                            elif abs(offset_x) > abs(offset_y):
                                self.current_zone = 'left' if offset_x < 0 else 'right'
                            else:
                                self.current_zone = 'up' if offset_y < 0 else 'down'
                            
                            # Get zone color based on fill status
                            zone_filled = self.zone_captures.get(self.current_zone, 0) >= self.zone_targets.get(self.current_zone, 0)
                            box_color = (0, 255, 0) if zone_filled else (0, 255, 255)
                            
                            # Draw face box
                            cv2.rectangle(display_frame, (int(left), int(top)), (int(right), int(bottom)), box_color, 3)
                            
                            # Auto-capture logic
                            now = time.time()
                            zone_current = self.zone_captures.get(self.current_zone, 0)
                            zone_target = self.zone_targets.get(self.current_zone, 0)
                            
                            if now - self.last_auto_capture >= self.auto_capture_interval:
                                if self.captured_count < self.auto_capture_target and zone_current < zone_target:
                                    filepath = self.face_system.save_face_image(frame, self.registration_name)
                                    self.captured_count += 1
                                    self.zone_captures[self.current_zone] = zone_current + 1
                                    self.last_auto_capture = now
                                    self.root.after(0, self.update_registration_ui)
                                elif self.captured_count >= self.auto_capture_target:
                                    self.root.after(0, self.complete_auto_registration)
                            
                            # Flash on capture
                            if (time.time() - self.last_auto_capture) < 0.08:
                                cv2.rectangle(display_frame, (0, 0), (frame_width, frame_height), (0, 255, 0), 12)
                        
                        elif len(faces) == 0:
                            cv2.putText(display_frame, "Position your face in frame", (frame_width // 2 - 150, frame_height // 2),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
                        else:
                            cv2.putText(display_frame, "Only one face please", (frame_width // 2 - 100, frame_height // 2),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Draw Face ID style overlay
                        self.draw_faceid_overlay(display_frame)
                    
                    elif self.is_training:
                        # Training in progress - just show clean feed, progress is in UI label
                        pass
                    else:
                        # Idle/manual capture mode - use cached async detection results
                        # (detection is already being submitted above for all registration modes)
                        if len(faces) == 1:
                            top, right, bottom, left = faces[0]
                            cv2.rectangle(display_frame, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), 2)
                        elif len(faces) == 0:
                            cv2.putText(display_frame, "Position face in frame", (frame_width // 2 - 120, frame_height // 2),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
                        else:
                            cv2.putText(display_frame, "Only one face please", (frame_width // 2 - 100, frame_height // 2),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                
                elif self.admin_mode:
                    # Admin mode but not in registration - show clean camera feed
                    # Just display the frame without overlays
                    pass
                
                # Periodic cache cleanup to prevent memory buildup
                now = time.time()
                if now - self.last_cache_cleanup > self.cache_cleanup_interval:
                    expired_count = self.face_system.face_cache.cleanup_expired()
                    if expired_count and expired_count > 0:
                        print(f"Cache cleanup: removed {expired_count} expired entries (cache size: {self.face_system.face_cache.size()})")
                    # Periodic garbage collection to reclaim memory from large frame arrays
                    gc.collect()
                    self.last_cache_cleanup = now
                
                # Store current frame reference (for registration capture)
                # Only copy if registration mode needs to save images
                if self.registration_mode:
                    self.current_frame = frame.copy() if self.auto_capture_mode else frame
                else:
                    self.current_frame = frame  # Reference only, no copy needed
                
                # Update display - pass display_frame which already has overlays drawn
                self.root.after(0, lambda f=display_frame.copy(): self.display_frame(f))
                
                # Adaptive frame rate - aim for ~30 FPS
                loop_time = time.time() - loop_start
                sleep_time = max(0.001, 0.033 - loop_time)
                time.sleep(sleep_time)
            
        except Exception as e:
            print(f"Camera error: {e}")
        finally:
            self.camera.stop()
    
    def display_frame(self, frame):
        """Display a frame on the video label (and admin preview if in admin mode)
        
        Optimized with:
        - Lazy BGR→RGB conversion (only when needed)
        - Pre-allocated resize buffer
        - Minimal object creation
        """
        frame_h, frame_w = frame.shape[:2]
        target_w, target_h = 440, 330
        
        # Resize to display size using pre-allocated buffer if needed
        if frame_w != target_w or frame_h != target_h:
            # Resize directly into pre-allocated buffer
            cv2.resize(frame, (target_w, target_h), dst=self._display_buffer)
            resized = self._display_buffer
        else:
            resized = frame
        
        # Lazy color conversion: BGR→RGB only for display
        if USE_PICAMERA:
            # Picamera already provides RGB
            frame_rgb = resized
        else:
            # Convert BGR to RGB using pre-allocated buffer
            cv2.cvtColor(resized, cv2.COLOR_BGR2RGB, dst=self._display_buffer_rgb)
            frame_rgb = self._display_buffer_rgb
        
        # Create PIL Image and PhotoImage
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Display on admin video label if admin panel is open
        if self.admin_mode and hasattr(self, 'admin_video_label'):
            try:
                if self.admin_video_label.winfo_exists():
                    self.admin_video_label.imgtk = imgtk
                    self.admin_video_label.configure(image=imgtk)
            except tk.TclError:
                pass  # Widget was destroyed
        else:
            # Update main video label
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
        login_dialog.geometry("350x280")
        login_dialog.configure(bg=Config.COLOR_BG)
        login_dialog.resizable(False, False)
        
        # Center on screen
        login_dialog.update_idletasks()
        x = (login_dialog.winfo_screenwidth() - 350) // 2
        y = (login_dialog.winfo_screenheight() - 280) // 2
        login_dialog.geometry(f"350x280+{x}+{y}")
        
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
        password_entry.bind('<Button-1>', lambda e: self._show_password_keyboard(login_dialog, content, password_entry))
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
        
        # Show keyboard immediately
        login_dialog.after(5, lambda: self._show_password_keyboard(login_dialog, content, password_entry))
        
        # Wait for dialog to close
        self.root.wait_window(login_dialog)
        
        # Check password
        password = result['password']
        if password == Config.ADMIN_PASSWORD:
            self.show_admin_panel()
        elif password is not None:
            messagebox.showerror("Error", "Invalid password")
    
    def _show_password_keyboard(self, dialog, content, entry):
        """Show keyboard in password dialog with expanded size"""
        # Expand dialog to fit keyboard
        dialog.geometry("480x450")
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() - 480) // 2
        y = (dialog.winfo_screenheight() - 450) // 2
        dialog.geometry(f"480x450+{x}+{y}")
        
        # Show keyboard
        show_keyboard(content, entry, dialog)
    
    def show_admin_panel(self):
        """Show the admin control panel - replaces kiosk UI in same window"""
        self.admin_mode = True
        self.is_scanning = False
        
        # Clear the face cache when entering admin mode
        self.face_system.clear_cache()
        
        # Hide the main kiosk UI
        self.main_frame.pack_forget()
        
        # Force window size to stay consistent
        if Config.FULLSCREEN:
            self.root.attributes('-fullscreen', True)
        else:
            self.root.geometry("480x800")
        
        # Create admin frame in the same window
        self.admin_frame = tk.Frame(self.root, bg=Config.COLOR_BG)
        self.admin_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title header
        header = tk.Frame(self.admin_frame, bg=Config.COLOR_BG)
        header.pack(fill=tk.X, padx=15, pady=(15, 8))
        
        tk.Label(
            header,
            text="Settings",
            font=(Config.FONT_FAMILY, 18, "bold"),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_BG
        ).pack(side=tk.LEFT)
        
        # Done button in header
        close_btn = tk.Button(
            header,
            text="Done",
            font=(Config.FONT_FAMILY, 12),
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
                       font=(Config.FONT_FAMILY, 9),
                       padding=[10, 6],
                       background=Config.COLOR_BG,
                       foreground=Config.COLOR_TEXT_SECONDARY)
        style.map('TNotebook.Tab',
                 background=[('selected', Config.COLOR_BG)],
                 foreground=[('selected', Config.COLOR_SCANNING)])
        
        notebook = ttk.Notebook(self.admin_frame)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        self.admin_notebook = notebook  # Store reference for tab locking
        
        # Bind tab change to check if allowed
        notebook.bind('<<NotebookTabChanged>>', self.on_tab_changed)
        
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
    
    def create_register_tab(self, parent):
        """Create registration tab in admin panel with camera preview"""
        # Main container - vertical layout
        self.reg_main_container = tk.Frame(parent, bg=Config.COLOR_BG)
        self.reg_main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Camera preview container with FIXED size to prevent expansion
        camera_container = tk.Frame(self.reg_main_container, bg=Config.COLOR_CARD, 
                                   highlightbackground=Config.COLOR_BORDER, highlightthickness=1,
                                   width=440, height=330)
        camera_container.pack(pady=(0, 5))
        camera_container.pack_propagate(False)  # Prevent size changes
        
        # Camera preview label
        self.admin_video_label = tk.Label(camera_container, bg="#000000")
        self.admin_video_label.pack(fill=tk.BOTH, expand=True)
        
        # === SETUP PANEL (shown before starting) ===
        self.reg_setup_panel = tk.Frame(self.reg_main_container, bg=Config.COLOR_CARD,
                                        highlightbackground=Config.COLOR_BORDER, highlightthickness=1)
        self.reg_setup_panel.pack(fill=tk.X, pady=3)
        
        setup_inner = tk.Frame(self.reg_setup_panel, bg=Config.COLOR_CARD)
        setup_inner.pack(fill=tk.X, padx=10, pady=8)
        
        # Name entry row
        name_row = tk.Frame(setup_inner, bg=Config.COLOR_CARD)
        name_row.pack(fill=tk.X)
        
        tk.Label(
            name_row,
            text="Name:",
            font=(Config.FONT_FAMILY, 10),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_CARD
        ).pack(side=tk.LEFT)
        
        self.reg_name_entry = tk.Entry(
            name_row, 
            font=(Config.FONT_FAMILY, 10), 
            bg=Config.COLOR_CARD_SECONDARY,
            fg=Config.COLOR_TEXT,
            relief=tk.FLAT,
            highlightbackground=Config.COLOR_BORDER,
            highlightthickness=1,
            width=30
        )
        self.reg_name_entry.pack(side=tk.LEFT, padx=(8, 10), ipady=3)
        self.reg_name_entry.bind('<Button-1>', lambda e: show_keyboard(self.reg_main_container, self.reg_name_entry, self.root))
        
        self.start_reg_btn = tk.Button(
            name_row,
            text="Start",
            font=(Config.FONT_FAMILY, 10),
            fg="#FFFFFF",
            bg=Config.COLOR_SCANNING,
            activebackground="#0056b3",
            activeforeground="#FFFFFF",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.start_registration
        )
        self.start_reg_btn.pack(side=tk.RIGHT, ipady=3, ipadx=10)
        
        # === CAPTURE PANEL (shown during registration) ===
        self.reg_capture_panel = tk.Frame(self.reg_main_container, bg=Config.COLOR_CARD,
                                          highlightbackground=Config.COLOR_BORDER, highlightthickness=1)
        # Don't pack yet - will be shown when registration starts
        
        capture_inner = tk.Frame(self.reg_capture_panel, bg=Config.COLOR_CARD)
        capture_inner.pack(fill=tk.X, padx=10, pady=8)
        
        # Top row: status and stop button
        top_row = tk.Frame(capture_inner, bg=Config.COLOR_CARD)
        top_row.pack(fill=tk.X, pady=(0, 5))
        
        self.reg_name_display = tk.Label(
            top_row,
            text="",
            font=(Config.FONT_FAMILY, 11, "bold"),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_CARD
        )
        self.reg_name_display.pack(side=tk.LEFT)
        
        self.reg_count_label = tk.Label(
            top_row,
            text="0 photos",
            font=(Config.FONT_FAMILY, 9),
            fg=Config.COLOR_TEXT_SECONDARY,
            bg=Config.COLOR_CARD
        )
        self.reg_count_label.pack(side=tk.LEFT, padx=(10, 0))
        
        self.stop_reg_btn = tk.Button(
            top_row,
            text="Stop",
            font=(Config.FONT_FAMILY, 9),
            fg="#FFFFFF",
            bg=Config.COLOR_DENIED,
            activeforeground="#FFFFFF",
            activebackground="#c0392b",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.stop_registration
        )
        self.stop_reg_btn.pack(side=tk.RIGHT, ipady=2, ipadx=8)
        
        # Bottom row: capture buttons
        btn_row = tk.Frame(capture_inner, bg=Config.COLOR_CARD)
        btn_row.pack(fill=tk.X)
        
        self.capture_btn = tk.Button(
            btn_row,
            text="📷 Capture",
            font=(Config.FONT_FAMILY, 9),
            fg="#FFFFFF",
            bg=Config.COLOR_GRANTED,
            activebackground="#28a745",
            activeforeground="#FFFFFF",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.capture_photo
        )
        self.capture_btn.pack(side=tk.LEFT, ipady=2, ipadx=8)
        
        self.auto_capture_btn = tk.Button(
            btn_row,
            text="⟳ Auto (100)",
            font=(Config.FONT_FAMILY, 9),
            fg="#FFFFFF",
            bg="#5856D6",
            activebackground="#4744c4",
            activeforeground="#FFFFFF",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.start_auto_capture
        )
        self.auto_capture_btn.pack(side=tk.LEFT, padx=(5, 0), ipady=2, ipadx=8)
        
        # Auto-train checkbox (smaller, right side)
        self.auto_train_var = tk.BooleanVar(value=True)
        self.auto_train_check = tk.Checkbutton(
            btn_row,
            text="Auto-train",
            variable=self.auto_train_var,
            font=(Config.FONT_FAMILY, 8),
            fg=Config.COLOR_TEXT_SECONDARY,
            bg=Config.COLOR_CARD,
            activebackground=Config.COLOR_CARD,
            selectcolor=Config.COLOR_CARD
        )
        self.auto_train_check.pack(side=tk.RIGHT)
    
    def create_train_tab(self, parent):
        """Create training tab in admin panel with Apple styling"""
        # Card container
        card = tk.Frame(parent, bg=Config.COLOR_CARD, highlightbackground=Config.COLOR_BORDER, highlightthickness=1)
        card.pack(fill=tk.X, padx=10, pady=10)
        
        inner = tk.Frame(card, bg=Config.COLOR_CARD)
        inner.pack(fill=tk.X, padx=15, pady=15)
        
        tk.Label(
            inner,
            text="Train Model",
            font=(Config.FONT_FAMILY, 13, "bold"),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_CARD
        ).pack(anchor=tk.W)
        
        tk.Label(
            inner,
            text="Process photos to train recognition",
            font=(Config.FONT_FAMILY, 9),
            fg=Config.COLOR_TEXT_SECONDARY,
            bg=Config.COLOR_CARD
        ).pack(anchor=tk.W, pady=(3, 15))
        
        # Dataset info
        persons = self.face_system.get_registered_persons()
        total_images = sum(count for _, count in persons)
        
        info_card = tk.Frame(inner, bg=Config.COLOR_CARD_SECONDARY)
        info_card.pack(fill=tk.X, pady=(0, 10))
        
        self.dataset_info_label = tk.Label(
            info_card,
            text=f"{len(persons)} people  •  {total_images} photos",
            font=(Config.FONT_FAMILY, 10),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_CARD_SECONDARY,
            pady=8
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
            length=300,
            style="Custom.Horizontal.TProgressbar"
        )
        self.train_progress.pack(fill=tk.X, pady=(0, 5))
        
        self.train_status_label = tk.Label(
            inner,
            text="Ready to train",
            font=(Config.FONT_FAMILY, 9),
            fg=Config.COLOR_TEXT_SECONDARY,
            bg=Config.COLOR_CARD
        )
        self.train_status_label.pack(pady=(0, 10))
        
        # Train button
        self.train_btn = tk.Button(
            inner,
            text="Start Training",
            font=(Config.FONT_FAMILY, 10),
            fg="#FFFFFF",
            bg=Config.COLOR_SCANNING,
            activebackground="#0056b3",
            activeforeground="#FFFFFF",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.start_training
        )
        self.train_btn.pack(fill=tk.X, ipady=6)
    
    def create_manage_tab(self, parent):
        """Create user management tab in admin panel with Apple styling"""
        # Header
        header = tk.Frame(parent, bg=Config.COLOR_BG)
        header.pack(fill=tk.X, padx=10, pady=(10, 5))
        
        tk.Label(
            header,
            text="Registered Users",
            font=(Config.FONT_FAMILY, 10, "bold"),
            fg=Config.COLOR_TEXT_SECONDARY,
            bg=Config.COLOR_BG
        ).pack(side=tk.LEFT)
        
        tk.Button(
            header,
            text="Refresh",
            font=(Config.FONT_FAMILY, 9),
            fg=Config.COLOR_SCANNING,
            bg=Config.COLOR_BG,
            activeforeground=Config.COLOR_SCANNING,
            bd=0,
            cursor="hand2",
            command=self.refresh_manage_list
        ).pack(side=tk.RIGHT)
        
        # List card
        card = tk.Frame(parent, bg=Config.COLOR_CARD, highlightbackground=Config.COLOR_BORDER, highlightthickness=1)
        card.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.manage_listbox = tk.Listbox(
            card,
            font=(Config.FONT_FAMILY, 10),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_CARD,
            selectbackground=Config.COLOR_CARD_SECONDARY,
            selectforeground=Config.COLOR_TEXT,
            highlightthickness=0,
            bd=0,
            relief=tk.FLAT,
            activestyle='none'
        )
        self.manage_listbox.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Populate list
        self.refresh_manage_list()
        
        # Delete button
        btn_frame = tk.Frame(parent, bg=Config.COLOR_BG)
        btn_frame.pack(fill=tk.X, padx=10, pady=8)
        
        tk.Button(
            btn_frame,
            text="Delete Selected",
            font=(Config.FONT_FAMILY, 9),
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
        header.pack(fill=tk.X, padx=10, pady=(10, 5))
        
        tk.Label(
            header,
            text="Access History",
            font=(Config.FONT_FAMILY, 10, "bold"),
            fg=Config.COLOR_TEXT_SECONDARY,
            bg=Config.COLOR_BG
        ).pack(side=tk.LEFT)
        
        tk.Button(
            header,
            text="Clear",
            font=(Config.FONT_FAMILY, 9),
            fg=Config.COLOR_DENIED,
            bg=Config.COLOR_BG,
            activeforeground=Config.COLOR_DENIED,
            bd=0,
            cursor="hand2",
            command=self.clear_access_log
        ).pack(side=tk.RIGHT)
        
        # Quick filter buttons
        filter_frame = tk.Frame(parent, bg=Config.COLOR_BG)
        filter_frame.pack(fill=tk.X, padx=10, pady=(0, 5))
        
        tk.Button(
            filter_frame,
            text="Today",
            font=(Config.FONT_FAMILY, 8),
            fg=Config.COLOR_SCANNING,
            bg=Config.COLOR_BG,
            activeforeground=Config.COLOR_SCANNING,
            bd=0,
            cursor="hand2",
            command=lambda: self.set_log_date_range(0)
        ).pack(side=tk.LEFT, padx=2)
        
        tk.Button(
            filter_frame,
            text="7 Days",
            font=(Config.FONT_FAMILY, 8),
            fg=Config.COLOR_SCANNING,
            bg=Config.COLOR_BG,
            activeforeground=Config.COLOR_SCANNING,
            bd=0,
            cursor="hand2",
            command=lambda: self.set_log_date_range(7)
        ).pack(side=tk.LEFT, padx=2)
        
        tk.Button(
            filter_frame,
            text="30 Days",
            font=(Config.FONT_FAMILY, 8),
            fg=Config.COLOR_SCANNING,
            bg=Config.COLOR_BG,
            activeforeground=Config.COLOR_SCANNING,
            bd=0,
            cursor="hand2",
            command=lambda: self.set_log_date_range(30)
        ).pack(side=tk.LEFT, padx=2)
        
        tk.Button(
            filter_frame,
            text="All",
            font=(Config.FONT_FAMILY, 8),
            fg=Config.COLOR_TEXT_SECONDARY,
            bg=Config.COLOR_BG,
            activeforeground=Config.COLOR_TEXT,
            bd=0,
            cursor="hand2",
            command=self.clear_log_filter
        ).pack(side=tk.LEFT, padx=2)
        
        # Log card
        card = tk.Frame(parent, bg=Config.COLOR_CARD, highlightbackground=Config.COLOR_BORDER, highlightthickness=1)
        card.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.admin_log_listbox = tk.Listbox(
            card,
            font=(Config.FONT_FAMILY_MONO, 9),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_CARD,
            selectbackground=Config.COLOR_CARD_SECONDARY,
            selectforeground=Config.COLOR_TEXT,
            highlightthickness=0,
            bd=0,
            relief=tk.FLAT,
            activestyle='none'
        )
        self.admin_log_listbox.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        
        # Initialize filter variables (needed by filter methods)
        self.log_date_from = None
        self.log_date_to = None
        self.log_name_var = tk.StringVar(value="All")
        
        # Populate with all entries initially
        self.populate_log_listbox(self.access_log.get_recent(100))
    
    def set_log_date_range(self, days_back):
        """Set date range for quick filters"""
        from datetime import timedelta, date
        today = datetime.now().date()
        
        if days_back == 0:
            date_from = today
        else:
            date_from = today - timedelta(days=days_back)
        
        # Get filtered entries
        entries = self.access_log.get_filtered(date_from, today, None, count=100)
        self.populate_log_listbox(entries)
    
    def clear_log_filter(self):
        """Clear all filters and show all entries"""
        self.log_name_var.set("All")
        self.populate_log_listbox(self.access_log.get_recent(100))
    
    def populate_log_listbox(self, entries):
        """Populate the log listbox with entries"""
        self.admin_log_listbox.delete(0, tk.END)
        for entry in entries:
            timestamp = datetime.fromisoformat(entry['timestamp']).strftime("%b %d, %H:%M:%S")
            status_icon = "●" if entry['access_granted'] else "○"
            self.admin_log_listbox.insert(tk.END, f"  {status_icon}  {timestamp}    {entry['name']}")
    
    def create_settings_tab(self, parent):
        """Create settings tab in admin panel with Apple styling"""
        # Recognition Settings Card
        card1 = tk.Frame(parent, bg=Config.COLOR_CARD, highlightbackground=Config.COLOR_BORDER, highlightthickness=1)
        card1.pack(fill=tk.X, padx=10, pady=(10, 5))
        
        inner1 = tk.Frame(card1, bg=Config.COLOR_CARD)
        inner1.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(
            inner1,
            text="Recognition",
            font=(Config.FONT_FAMILY, 12, "bold"),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_CARD
        ).pack(anchor=tk.W)
        
        settings_items = [
            ("Confidence", f"{int(Config.RECOGNITION_THRESHOLD * 100)}%"),
            ("Cooldown", f"{Config.COOLDOWN_SECONDS}s"),
            ("Unlock Duration", f"{Config.DOOR_UNLOCK_DURATION}s"),
        ]
        
        for label, value in settings_items:
            row = tk.Frame(inner1, bg=Config.COLOR_CARD)
            row.pack(fill=tk.X, pady=4)
            tk.Label(
                row,
                text=label,
                font=(Config.FONT_FAMILY, 10),
                fg=Config.COLOR_TEXT,
                bg=Config.COLOR_CARD
            ).pack(side=tk.LEFT)
            tk.Label(
                row,
                text=value,
                font=(Config.FONT_FAMILY, 10),
                fg=Config.COLOR_TEXT_SECONDARY,
                bg=Config.COLOR_CARD
            ).pack(side=tk.RIGHT)
        
        # System Info Card
        card2 = tk.Frame(parent, bg=Config.COLOR_CARD, highlightbackground=Config.COLOR_BORDER, highlightthickness=1)
        card2.pack(fill=tk.X, padx=10, pady=5)
        
        inner2 = tk.Frame(card2, bg=Config.COLOR_CARD)
        inner2.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Label(
            inner2,
            text="System",
            font=(Config.FONT_FAMILY, 12, "bold"),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_CARD
        ).pack(anchor=tk.W)
        
        camera_type = "Pi Camera" if USE_PICAMERA else "USB Webcam"
        gpio_status = "Hardware" if USE_GPIO else "Simulated"
        
        system_items = [
            ("Camera", camera_type),
            ("Door Control", gpio_status),
            ("Performance", "Optimized" if Config.USE_FAST_DETECTION else "Standard"),
        ]
        
        for label, value in system_items:
            row = tk.Frame(inner2, bg=Config.COLOR_CARD)
            row.pack(fill=tk.X, pady=4)
            tk.Label(
                row,
                text=label,
                font=(Config.FONT_FAMILY, 10),
                fg=Config.COLOR_TEXT,
                bg=Config.COLOR_CARD
            ).pack(side=tk.LEFT)
            tk.Label(
                row,
                text=value,
                font=(Config.FONT_FAMILY, 10),
                fg=Config.COLOR_TEXT_SECONDARY,
                bg=Config.COLOR_CARD
            ).pack(side=tk.RIGHT)
        
        # Exit button
        exit_frame = tk.Frame(parent, bg=Config.COLOR_BG)
        exit_frame.pack(fill=tk.X, padx=10, pady=15)
        
        tk.Button(
            exit_frame,
            text="Exit Kiosk Mode",
            font=(Config.FONT_FAMILY, 10),
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
    
    def refresh_train_tab(self):
        """Refresh the train tab dataset info"""
        if hasattr(self, 'dataset_info_label'):
            persons = self.face_system.get_registered_persons()
            total_images = sum(count for _, count in persons)
            self.dataset_info_label.config(text=f"{len(persons)} people  •  {total_images} photos")
    
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
        self.zone_captures = {'center': 0, 'left': 0, 'right': 0, 'up': 0, 'down': 0}
        
        # Swap panels - hide setup, show capture
        self.reg_setup_panel.pack_forget()
        self.reg_capture_panel.pack(fill=tk.X, pady=5)
        
        # Update capture panel with name
        self.reg_name_display.config(text=name)
        self.reg_count_label.config(text="0 photos")
    
    def capture_photo(self):
        """Capture a photo for registration"""
        if self.current_frame is not None and self.registration_mode:
            filepath = self.face_system.save_face_image(self.current_frame, self.registration_name)
            self.captured_count += 1
            self.reg_count_label.config(text=f"{self.captured_count} photos")
            print(f"[REGISTER] Saved: {filepath}")
    
    def stop_registration(self):
        """Stop face registration mode and optionally auto-train"""
        # Prevent stopping during capture or training
        if self.reg_process_locked:
            messagebox.showwarning("Please Wait", "Capture or encoding in progress. Please wait for completion.")
            return
        
        person_name = self.registration_name
        captured = self.captured_count
        
        self.registration_mode = False
        self.registration_name = ""
        self.auto_capture_mode = False
        self.zone_captures = {'center': 0, 'left': 0, 'right': 0, 'up': 0, 'down': 0}
        
        # Swap panels - hide capture, show setup
        self.reg_capture_panel.pack_forget()
        self.reg_setup_panel.pack(fill=tk.X, pady=5)
        
        # Reset capture panel buttons for next time
        self.auto_capture_btn.config(text="⟳ Auto Capture (100 photos)", bg="#5856D6", command=self.start_auto_capture)
        self.capture_btn.config(state=tk.NORMAL)
        
        # Clear name entry
        self.reg_name_entry.delete(0, tk.END)
        
        self.refresh_manage_list()
        self.refresh_train_tab()
        
        # Auto-train the new person if option is enabled and photos were captured
        # Skip if already trained via auto-capture flow
        already_trained = getattr(self, 'already_trained', False)
        self.already_trained = False  # Reset for next registration
        if not already_trained and hasattr(self, 'auto_train_var') and self.auto_train_var.get() and captured > 0 and person_name:
            self.train_single_person(person_name)
    
    # ==================== AUTO-CAPTURE FACE ID STYLE ====================
    
    def start_auto_capture(self):
        """Start automatic Face ID style capture"""
        if not self.registration_mode:
            return
        
        self.auto_capture_mode = True
        self.reg_process_locked = True  # Lock navigation until everything is done
        self.last_auto_capture = time.time()
        self.zone_captures = {'center': 0, 'left': 0, 'right': 0, 'up': 0, 'down': 0}
        
        # Update UI
        self.capture_btn.config(state=tk.DISABLED)
        self.auto_capture_btn.config(text="⏹ Stop", bg=Config.COLOR_DENIED, command=self.stop_auto_capture)
        self.reg_count_label.config(text="Move face to fill all zones...")
    
    def stop_auto_capture(self):
        """Stop automatic capture - requires confirmation"""
        if self.reg_process_locked:
            # Ask for confirmation to cancel
            if not messagebox.askyesno("Cancel Capture?", 
                "Are you sure you want to cancel?\nPhotos already captured will be kept but training won't start."):
                return
        
        self.auto_capture_mode = False
        self.reg_process_locked = False
        self.capture_btn.config(state=tk.NORMAL)
        self.auto_capture_btn.config(text="⟳ Auto Capture (100 photos)", bg="#5856D6", command=self.start_auto_capture)
        self.update_registration_ui()
    
    def draw_faceid_overlay(self, frame):
        """Draw Face ID style visual guide with zone indicators"""
        frame_height, frame_width = frame.shape[:2]
        center_x, center_y = frame_width // 2, frame_height // 2
        
        # Draw circular Face ID style guide in center
        guide_radius = min(frame_width, frame_height) // 4
        
        # Outer circle (face positioning guide)
        cv2.circle(frame, (center_x, center_y), guide_radius, (100, 100, 100), 2)
        
        # Zone indicator positions (around the circle)
        zone_positions = {
            'up': (center_x, center_y - guide_radius - 30),
            'down': (center_x, center_y + guide_radius + 30),
            'left': (center_x - guide_radius - 30, center_y),
            'right': (center_x + guide_radius + 30, center_y),
            'center': (center_x, center_y)
        }
        
        # Draw zone indicators
        for zone, (zx, zy) in zone_positions.items():
            current = self.zone_captures.get(zone, 0)
            target = self.zone_targets.get(zone, 0)
            
            # Calculate fill percentage
            fill_pct = min(1.0, current / target) if target > 0 else 1.0
            
            # Colors: gray unfilled, yellow partial, green complete
            if fill_pct >= 1.0:
                color = (0, 255, 0)  # Green - complete
            elif fill_pct > 0:
                color = (0, 255, 255)  # Yellow - partial
            else:
                color = (80, 80, 80)  # Gray - empty
            
            # Highlight current zone
            radius = 18 if zone == self.current_zone else 12
            thickness = -1 if fill_pct >= 1.0 else 2
            
            if zone == 'center':
                # Center is a ring, not a dot
                cv2.circle(frame, (zx, zy), radius + 5, color, 2)
            else:
                cv2.circle(frame, (zx, zy), radius, color, thickness)
                # Show count
                if fill_pct < 1.0:
                    cv2.putText(frame, str(current), (zx - 8, zy + 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Header
        mode_text = "FACE ID CAPTURE" if self.auto_capture_mode else "REGISTRATION"
        cv2.putText(frame, mode_text, (15, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 100, 200), 2)
        cv2.putText(frame, self.registration_name, (15, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Progress bar at bottom
        if self.auto_capture_mode:
            bar_width = 350
            bar_height = 14
            bar_x = center_x - bar_width // 2
            bar_y = frame_height - 45
            
            progress = self.captured_count / self.auto_capture_target
            
            # Background
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (40, 40, 40), -1)
            # Fill
            fill_width = int(bar_width * progress)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), (0, 200, 0), -1)
            # Border
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), 1)
            
            # Count text
            count_text = f"{self.captured_count}/{self.auto_capture_target}"
            cv2.putText(frame, count_text, (bar_x + bar_width + 10, bar_y + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Guidance text
            zones_complete = sum(1 for z in self.zone_captures if self.zone_captures[z] >= self.zone_targets.get(z, 0))
            if zones_complete < 5:
                # Find which zone needs photos
                for zone in ['center', 'left', 'right', 'up', 'down']:
                    if self.zone_captures.get(zone, 0) < self.zone_targets.get(zone, 0):
                        guidance = {
                            'center': "Look at the center",
                            'left': "Move face LEFT",
                            'right': "Move face RIGHT", 
                            'up': "Move face UP",
                            'down': "Move face DOWN"
                        }
                        cv2.putText(frame, guidance.get(zone, ""), (center_x - 80, bar_y - 15),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                        break
            else:
                cv2.putText(frame, "All angles captured!", (center_x - 90, bar_y - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    
    def update_registration_ui(self):
        """Update the registration UI with current progress"""
        if self.auto_capture_mode:
            zones_done = sum(1 for z in self.zone_captures 
                           if self.zone_captures[z] >= self.zone_targets.get(z, 0))
            self.reg_count_label.config(
                text=f"{self.captured_count} photos • {zones_done}/5 zones"
            )
        else:
            self.reg_count_label.config(text=f"{self.captured_count} photos")
    
    def complete_auto_registration(self):
        """Called when auto-capture reaches target - automatically start training"""
        self.auto_capture_mode = False
        # reg_process_locked stays True - will be cleared after encoding completes
        zones_done = sum(1 for z in self.zone_captures 
                        if self.zone_captures[z] >= self.zone_targets.get(z, 0))
        
        self.reg_count_label.config(
            text=f"✓ {self.captured_count} photos • Training..."
        )
        
        # Disable buttons during training
        self.capture_btn.config(state=tk.DISABLED)
        self.auto_capture_btn.config(text="Encoding...", bg="#888888", state=tk.DISABLED)
        self.stop_reg_btn.config(state=tk.DISABLED)
        
        # Start training with progress (lock already set from start_auto_capture)
        self.train_single_person_with_progress(self.registration_name)
    
    # ==================== END AUTO-CAPTURE ====================
    
    def train_single_person_with_progress(self, person_name):
        """Train a single person with progress updates in the UI"""
        self.is_training = True
        self.reg_process_locked = True  # Ensure lock is set
        total_images = 0
        
        def update_progress_label(text):
            """Safely update progress label"""
            try:
                if hasattr(self, 'reg_count_label') and self.reg_count_label.winfo_exists():
                    self.reg_count_label.config(text=text)
            except tk.TclError:
                pass
        
        def training_thread():
            nonlocal total_images
            # Count images first
            from imutils import paths
            person_folder = os.path.join(self.face_system.dataset_path, person_name)
            if os.path.exists(person_folder):
                total_images = len(list(paths.list_images(person_folder)))
            
            def progress_callback(current, total, filepath):
                if current == total:
                    # Last image - show saving message
                    self.root.after(0, lambda: update_progress_label("Saving encodings..."))
                else:
                    self.root.after(0, lambda c=current, t=total: update_progress_label(f"Encoding {c}/{t}..."))
            
            success, message = self.face_system.train_single_person(person_name, progress_callback)
            # Only release lock after training AND saving is complete
            self.root.after(0, lambda: self.training_with_progress_complete(success, message))
        
        thread = threading.Thread(target=training_thread, daemon=True)
        thread.start()
    
    def training_with_progress_complete(self, success, message):
        """Handle training completion from auto-capture flow"""
        self.is_training = False
        self.reg_process_locked = False  # NOW we can release the lock - everything is done
        
        # Safely update UI if widgets still exist
        try:
            if hasattr(self, 'reg_count_label') and self.reg_count_label.winfo_exists():
                if success:
                    self.reg_count_label.config(text=f"✓ {message}")
                else:
                    self.reg_count_label.config(text=f"✗ {message}")
            
            if hasattr(self, 'auto_capture_btn') and self.auto_capture_btn.winfo_exists():
                if success:
                    self.auto_capture_btn.config(text="✓ Complete", bg=Config.COLOR_GRANTED)
                else:
                    self.auto_capture_btn.config(text="✗ Failed", bg=Config.COLOR_DENIED)
            
            if not success and hasattr(self, 'stop_reg_btn') and self.stop_reg_btn.winfo_exists():
                self.stop_reg_btn.config(state=tk.NORMAL)
            
            if success:
                # Set flag to prevent double-training in stop_registration
                self.already_trained = True
                # Auto-close registration after short delay
                self.root.after(1500, self.stop_registration)
        except tk.TclError:
            pass  # Widgets were destroyed
    
    def train_single_person(self, person_name):
        """Train only a single person (incremental training)"""
        # Safely update label if it exists
        try:
            if hasattr(self, 'reg_count_label') and self.reg_count_label.winfo_exists():
                self.reg_count_label.config(text=f"Training {person_name}...")
        except tk.TclError:
            pass
        
        self.is_training = True  # Pause recognition during training
        
        def training_thread():
            success, message = self.face_system.train_single_person(person_name)
            self.root.after(0, lambda: self.single_training_complete(success, message))
        
        thread = threading.Thread(target=training_thread, daemon=True)
        thread.start()
    
    def single_training_complete(self, success, message):
        """Handle single person training completion"""
        self.is_training = False  # Resume recognition
        
        # Safely update label if it exists
        try:
            if hasattr(self, 'reg_count_label') and self.reg_count_label.winfo_exists():
                if success:
                    self.reg_count_label.config(text=f"✓ {message}")
                else:
                    self.reg_count_label.config(text=f"✗ Training failed")
                # Reset after a delay
                self.root.after(3000, self._reset_reg_count_label)
        except tk.TclError:
            pass
        
        if success:
            messagebox.showinfo("Training Complete", message)
        else:
            messagebox.showerror("Training Failed", message)
    
    def _reset_reg_count_label(self):
        """Safely reset the registration count label"""
        try:
            if hasattr(self, 'reg_count_label') and self.reg_count_label.winfo_exists():
                self.reg_count_label.config(text="0 photos")
        except tk.TclError:
            pass
    
    def start_training(self):
        """Start model training"""
        self.train_btn.config(state=tk.DISABLED)
        self.train_progress['value'] = 0
        self.train_status_label.config(text="Training in progress...")
        self.is_training = True  # Pause recognition during training
        
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
        self.is_training = False  # Resume recognition
        self.train_btn.config(state=tk.NORMAL)
        self.train_progress['value'] = 100 if success else 0
        self.train_status_label.config(text=message)
        
        if success:
            messagebox.showinfo("Training Complete", message)
            self.update_info_label()
        else:
            messagebox.showerror("Training Failed", message)
    
    def delete_person(self):
        """Delete selected person from dataset and model"""
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
        
        if messagebox.askyesno("Confirm Delete", f"Delete '{name}' from dataset and recognition model?"):
            import shutil
            
            # Delete from dataset folder
            person_folder = os.path.join(self.face_system.dataset_path, name)
            if os.path.exists(person_folder):
                shutil.rmtree(person_folder)
            
            # Remove from trained model
            success, message = self.face_system.remove_person_from_model(name)
            if success:
                print(f"[DELETE] {message}")
            else:
                print(f"[DELETE] Warning: {message}")
            
            # Update UI
            self.refresh_manage_list()
            self.update_info_label()
            
            messagebox.showinfo("Deleted", f"'{name}' has been removed from the system.")
    
    def clear_access_log(self):
        """Clear the access log"""
        if messagebox.askyesno("Confirm", "Clear all access log entries?"):
            self.access_log.clear()
            self.admin_log_listbox.delete(0, tk.END)
            self.update_log_display()
    
    def update_info_label(self):
        """Update the info label"""
        count = len(self.face_system.get_trained_persons())
        self.info_label.config(text=f"{count} Users")
    
    def on_tab_changed(self, event):
        """Handle notebook tab change - prevent if capture or training in progress"""
        if self.reg_process_locked:
            # Force back to register tab
            self.admin_notebook.select(0)
            messagebox.showwarning("Process in Progress", "Please wait for capture and encoding to complete.")
    
    def close_admin_panel(self):
        """Close the admin panel and restore kiosk UI"""
        # Prevent closing during capture or training
        if self.reg_process_locked:
            messagebox.showwarning("Process in Progress", "Please wait for capture and encoding to complete.")
            return
        
        if self.registration_mode:
            self.stop_registration()
        
        self.admin_mode = False
        self.is_scanning = True
        
        # Clear cache and reset performance stats when returning to scanning
        self.face_system.clear_cache()
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Destroy admin frame and restore main kiosk UI
        self.admin_frame.destroy()
        
        # Force window size to stay consistent
        if Config.FULLSCREEN:
            self.root.attributes('-fullscreen', True)
        else:
            self.root.geometry("480x800")
        
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
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
        
        # Stop recognition worker first
        if hasattr(self, 'recognition_worker'):
            self.recognition_worker.stop()
        
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
