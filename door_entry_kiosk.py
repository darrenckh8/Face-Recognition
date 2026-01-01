# ==================== STANDARD LIBRARY IMPORTS ====================
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import os
import threading
from queue import Queue, Empty
import time
from datetime import datetime
import pickle
import numpy as np
from PIL import Image, ImageTk
import json
import gc
import hashlib
import logging
import shutil
from typing import Optional, List, Tuple, Dict, Any

# ==================== LOGGING CONFIGURATION ====================
# Configure application-wide logging with timestamp and level information
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('DoorEntry')

# ==================== OPTIONAL DEPENDENCIES ====================
# InsightFace: Required for face detection and recognition
try:
    from insightface.app import FaceAnalysis
    import insightface
except ImportError:
    logger.error("insightface library not found. Please install it with: pip install insightface onnxruntime")
    exit(1)

# Picamera2: Optional - enables Raspberry Pi camera support
USE_PICAMERA = False
try:
    from picamera2 import Picamera2
    USE_PICAMERA = True
    logger.info("Picamera2 available - Raspberry Pi camera support enabled")
except ImportError:
    USE_PICAMERA = False

# RPi.GPIO: Optional - enables physical door relay control on Raspberry Pi
USE_GPIO = False
try:
    import RPi.GPIO as GPIO
    USE_GPIO = True
    logger.info("RPi.GPIO available - Door control enabled")
except ImportError:
    USE_GPIO = False


# ==================== SECURITY UTILITIES ====================
def hash_password(password: str) -> str:
    """
    Hash a password using SHA-256 with a salt for secure storage.
    
    Args:
        password: The plaintext password to hash
        
    Returns:
        The hexadecimal SHA-256 hash of the salted password
        
    Note:
        In production, use a unique random salt per user stored alongside the hash.
    """
    salt = "door_entry_kiosk_2024"
    return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()

def verify_password(password: str, hashed: str) -> bool:
    """
    Verify a plaintext password against a stored hash.
    
    Args:
        password: The plaintext password to verify
        hashed: The stored hash to compare against
        
    Returns:
        True if the password matches, False otherwise
    """
    return hash_password(password) == hashed


# ==================== APPLICATION CONFIGURATION ====================
class Config:
    """Central configuration class for all application settings.
    
    Modify these values to customize the kiosk behavior.
    """
    
    # ----- Window Settings -----
    FULLSCREEN = False                    # Set True for production kiosk deployment
    WINDOW_TITLE = "Door Entry System"    # Window title bar text
    
    # ----- Security Settings -----
    # Default admin password hash (plaintext: "admin123") - CHANGE IN PRODUCTION!
    ADMIN_PASSWORD_HASH = hash_password("admin123")
    
    # ----- Camera Settings -----
    CAMERA_RESOLUTION = (640, 480)        # Camera capture resolution (width, height)
    
    # ----- Face Recognition Settings -----
    RECOGNITION_THRESHOLD = 0.8          # Minimum cosine similarity for a match
    COOLDOWN_SECONDS = 5                  # Seconds between access logs for same person
    
    # ----- Performance Tuning -----
    RECOGNITION_INTERVAL_FRAMES = 2       # Process every Nth frame for recognition
    FACE_CACHE_TTL = 5.0                  # Seconds to remember a recognized face
    FACE_POSITION_TOLERANCE = 100         # Pixel tolerance for face position matching
    DETECTION_SCALE_FACTOR = 4            # Downscale factor for Haar cascade detection
    USE_FAST_DETECTION = False            # Use Haar cascade (faster but less accurate)
    EMBEDDING_CACHE_THRESHOLD = 0.6       # Similarity threshold for cache matching
    
    # ----- Memory Management Settings -----
    MAX_CACHE_ENTRIES = 20  # Maximum faces to keep in cache
    ACCESS_LOG_MAX_MEMORY_ENTRIES = 1000  # Max entries to keep in memory (older are on disk only)
    FRAME_POOL_SIZE = 3  # Number of pre-allocated frames for recognition queue
    
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
    COLOR_SCANNING = "#007AFF"                # Blue - face detected, processing
    COLOR_WARNING = "#FF9500"                 # Orange - warnings/alerts
    
    # ----- Light Theme Colors -----
    COLOR_BG = "#F2F2F7"                       # Main window background
    COLOR_CARD = "#FFFFFF"                    # Card/panel background
    COLOR_CARD_SECONDARY = "#F9F9F9"          # Alternate card background
    COLOR_TEXT = "#1C1C1E"                    # Primary text color
    COLOR_TEXT_SECONDARY = "#8E8E93"          # Secondary/muted text
    COLOR_TEXT_TERTIARY = "#AEAEB2"           # Tertiary/disabled text
    COLOR_BORDER = "#E5E5EA"                  # Border color for cards/inputs
    COLOR_SHADOW = "#C7C7CC"                  # Drop shadow color
    COLOR_HIGHLIGHT = "#E3F2FD"               # Hover/focus highlight
    
    # ----- UI Animation Timing (milliseconds) -----
    STATUS_DISPLAY_DURATION = 3000            # How long to show access granted/denied
    ANIMATION_DURATION = 150                  # Button/transition animations
    TOAST_DURATION = 2500                     # Toast notification display time
    
    # ----- Typography -----
    # Uses platform-native fonts for best appearance
    FONT_FAMILY = "SF Pro Display" if os.name == 'darwin' else "Segoe UI" if os.name == 'nt' else "Helvetica Neue"
    FONT_FAMILY_MONO = "SF Mono" if os.name == 'darwin' else "Consolas" if os.name == 'nt' else "Monaco"
    
    # ----- Backup Configuration -----
    BACKUP_ENABLED = True                     # Auto-backup before encoding changes
    MAX_BACKUPS = 5                           # Rolling backup limit (oldest deleted)
    
    @classmethod
    def validate(cls) -> List[str]:
        """Validate configuration values and return list of errors"""
        errors = []
        
        # Validate thresholds
        if not 0.0 <= cls.RECOGNITION_THRESHOLD <= 1.0:
            errors.append(f"RECOGNITION_THRESHOLD must be between 0.0 and 1.0, got {cls.RECOGNITION_THRESHOLD}")
        if not 0.0 <= cls.EMBEDDING_CACHE_THRESHOLD <= 1.0:
            errors.append(f"EMBEDDING_CACHE_THRESHOLD must be between 0.0 and 1.0, got {cls.EMBEDDING_CACHE_THRESHOLD}")
        
        # Validate positive integers
        if cls.COOLDOWN_SECONDS < 0:
            errors.append(f"COOLDOWN_SECONDS must be non-negative, got {cls.COOLDOWN_SECONDS}")
        if cls.MAX_CACHE_ENTRIES < 1:
            errors.append(f"MAX_CACHE_ENTRIES must be at least 1, got {cls.MAX_CACHE_ENTRIES}")
        if cls.ACCESS_LOG_MAX_MEMORY_ENTRIES < 1:
            errors.append(f"ACCESS_LOG_MAX_MEMORY_ENTRIES must be at least 1, got {cls.ACCESS_LOG_MAX_MEMORY_ENTRIES}")
        
        # Validate camera resolution
        if cls.CAMERA_RESOLUTION[0] < 320 or cls.CAMERA_RESOLUTION[1] < 240:
            errors.append(f"CAMERA_RESOLUTION too small: {cls.CAMERA_RESOLUTION}")
        
        # Validate file paths are writable
        for path_name in ['DATASET_PATH', 'ENCODINGS_PATH', 'ACCESS_LOG_PATH']:
            path = getattr(cls, path_name)
            parent_dir = os.path.dirname(path) or '.'
            if not os.access(parent_dir, os.W_OK):
                errors.append(f"{path_name} parent directory is not writable: {parent_dir}")
        
        return errors


# ==================== BACKUP SYSTEM ====================
class BackupManager:
    """
    Handles automatic backup of critical data files (encodings.pickle).
    
    Creates timestamped backups before any modification and maintains
    a rolling window of recent backups for recovery purposes.
    """
    
    @staticmethod
    def create_backup(file_path: str, max_backups: int = 5) -> Optional[str]:
        """
        Create a timestamped backup copy of a file.
        
        Args:
            file_path: Path to the file to backup
            max_backups: Maximum number of backups to retain
            
        Returns:
            Path to the created backup file, or None if backup failed
        """
        if not os.path.exists(file_path):
            return None
        
        try:
            # Generate unique backup filename with current timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base, ext = os.path.splitext(file_path)
            backup_path = f"{base}_backup_{timestamp}{ext}"
            
            # Preserve file metadata during copy
            shutil.copy2(file_path, backup_path)
            logger.info(f"Created backup: {backup_path}")
            
            # Enforce backup retention limit
            BackupManager._cleanup_old_backups(base, ext, max_backups)
            
            return backup_path
        except Exception as e:
            logger.error(f"Failed to create backup of {file_path}: {e}")
            return None
    
    @staticmethod
    def _cleanup_old_backups(base: str, ext: str, max_backups: int):
        """
        Remove excess backup files beyond the retention limit.
        Keeps the most recent backups based on filename timestamp.
        """
        import glob
        pattern = f"{base}_backup_*{ext}"
        backups = sorted(glob.glob(pattern), reverse=True)  # Newest first
        
        # Delete all backups beyond the limit
        for old_backup in backups[max_backups:]:
            try:
                os.remove(old_backup)
                logger.debug(f"Removed old backup: {old_backup}")
            except Exception as e:
                logger.warning(f"Failed to remove old backup {old_backup}: {e}")
    
    @staticmethod
    def get_latest_backup(file_path: str) -> Optional[str]:
        """
        Find the most recent backup file for a given source file.
        
        Args:
            file_path: Path to the original file
            
        Returns:
            Path to the newest backup, or None if no backups exist
        """
        import glob
        base, ext = os.path.splitext(file_path)
        pattern = f"{base}_backup_*{ext}"
        backups = sorted(glob.glob(pattern), reverse=True)
        return backups[0] if backups else None
    
    @staticmethod
    def restore_from_backup(file_path: str, backup_path: str = None) -> bool:
        """
        Restore a file from a backup copy.
        
        Args:
            file_path: Destination path to restore to
            backup_path: Specific backup to use (defaults to most recent)
            
        Returns:
            True if restore succeeded, False otherwise
        """
        if backup_path is None:
            backup_path = BackupManager.get_latest_backup(file_path)
        
        if backup_path is None or not os.path.exists(backup_path):
            logger.error(f"No backup found for {file_path}")
            return False
        
        try:
            shutil.copy2(backup_path, file_path)
            logger.info(f"Restored {file_path} from {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to restore from backup: {e}")
            return False


# ==================== FACE STABILITY TRACKER ====================
class FaceStabilityTracker:
    """
    Tracks face movement over consecutive frames to detect stability.
    
    This prevents recognition from running on moving faces, which can
    cause inconsistent results. Recognition only triggers when a face
    has remained relatively stationary for several frames.
    
    Attributes:
        stability_threshold: Maximum pixel movement allowed to be considered stable
        stable_frames_required: Number of consecutive stable frames needed
    """
    
    def __init__(self, stability_threshold: int = 300, stable_frames_required: int = 2):
        """
        Initialize the stability tracker.
        
        Args:
            stability_threshold: Max pixel distance between frames to count as stable
            stable_frames_required: How many stable frames before recognition triggers
        """
        self.stability_threshold = stability_threshold
        self.stable_frames_required = stable_frames_required
        self.last_positions: Dict[int, List[Tuple[int, int]]] = {}  # Track position history per face
        self.stable_count: Dict[int, int] = {}  # Consecutive stable frames per face
        self.lock = threading.Lock()  # Thread-safe access
    
    def _get_face_center(self, location: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """Calculate the center point of a face bounding box."""
        top, right, bottom, left = location
        return ((left + right) // 2, (top + bottom) // 2)
    
    def _calculate_movement(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate Euclidean distance between two points."""
        return ((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2) ** 0.5
    
    def update_and_check_stability(self, face_id: int, location: Tuple[int, int, int, int]) -> bool:
        """
        Update tracking for a face and check if it's stable.
        
        Args:
            face_id: Unique identifier for the face being tracked
            location: Bounding box as (top, right, bottom, left)
            
        Returns:
            True if face has been stable for required number of frames
        """
        center = self._get_face_center(location)
        
        with self.lock:
            # First time seeing this face
            if face_id not in self.last_positions:
                self.last_positions[face_id] = [center]
                self.stable_count[face_id] = 0
                return False
            
            positions = self.last_positions[face_id]
            
            # Compare to last known position
            if positions:
                movement = self._calculate_movement(center, positions[-1])
                
                if movement <= self.stability_threshold:
                    # Face is stable, increment counter
                    self.stable_count[face_id] = self.stable_count.get(face_id, 0) + 1
                else:
                    # Face moved too much, reset stability tracking
                    self.stable_count[face_id] = 0
            
            # Maintain rolling window of recent positions
            positions.append(center)
            if len(positions) > 10:
                positions.pop(0)
            
            return self.stable_count[face_id] >= self.stable_frames_required
    
    def clear(self):
        """Clear all tracking data for all faces."""
        with self.lock:
            self.last_positions.clear()
            self.stable_count.clear()
    
    def remove_face(self, face_id: int):
        """Remove tracking data for a specific face when it leaves view."""
        with self.lock:
            self.last_positions.pop(face_id, None)
            self.stable_count.pop(face_id, None)


# ==================== FACE CACHE ====================
class FaceCache:
    """
    Caches recognized faces to prevent redundant recognition processing.
    
    Uses a dual-matching strategy combining:
    - Position-based matching: Grid cells for approximate location matching
    - Embedding similarity: Cosine similarity for identity verification
    
    This allows the system to skip recognition for faces that have already
    been identified and are still visible in the camera view.
    """
    
    def __init__(self, ttl: Optional[float] = None, position_tolerance: Optional[int] = None, 
                 max_entries: Optional[int] = None):
        """
        Initialize the face cache.
        
        Args:
            ttl: Time-to-live in seconds before cache entries expire
            position_tolerance: Pixel tolerance for position-based matching
            max_entries: Maximum number of faces to cache simultaneously
        """
        self.ttl = ttl or Config.FACE_CACHE_TTL
        self.position_tolerance = position_tolerance or Config.FACE_POSITION_TOLERANCE
        self.embedding_threshold = getattr(Config, 'EMBEDDING_CACHE_THRESHOLD', 0.6)
        self.max_entries = max_entries or getattr(Config, 'MAX_CACHE_ENTRIES', 20)
        self.cache: Dict[Tuple[int, int], Dict[str, Any]] = {}
        self.lock = threading.Lock()
    
    def _get_position_key(self, location: Tuple[int, int, int, int]) -> Tuple[int, int]:
        """
        Convert face bounding box to a grid cell key for fast lookup.
        
        Divides the frame into grid cells based on position_tolerance,
        allowing efficient cache lookups for faces in similar positions.
        """
        top, right, bottom, left = location
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2
        grid_x = center_x // self.position_tolerance
        grid_y = center_y // self.position_tolerance
        return (grid_x, grid_y)
    
    def _compute_embedding_similarity(self, enc1: np.ndarray, enc2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two face embeddings.
        
        Returns a value between -1 and 1, where 1 indicates identical faces.
        Used to determine if a cached face matches the current face.
        """
        if enc1 is None or enc2 is None:
            return 0.0
        norm1 = np.linalg.norm(enc1)
        norm2 = np.linalg.norm(enc2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(enc1, enc2) / (norm1 * norm2))
    
    def _find_nearby_cache(self, location: Tuple[int, int, int, int], 
                           encoding: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
        """
        Search for a cached face matching the given location or embedding.
        
        Searches all non-expired cache entries and returns the best match
        based on a combined score of position proximity and embedding similarity.
        """
        top, right, bottom, left = location
        center_x = (left + right) // 2
        center_y = (top + bottom) // 2
        
        now = time.time()
        best_match = None
        best_score = -1
        
        with self.lock:
            # Collect expired entries for cleanup
            expired_keys = []
            for key, entry in self.cache.items():
                if now - entry['timestamp'] > self.ttl:
                    expired_keys.append(key)
                    continue
                
                # Calculate position distance
                cached_top, cached_right, cached_bottom, cached_left = entry['location']
                cached_center_x = (cached_left + cached_right) // 2
                cached_center_y = (cached_top + cached_bottom) // 2
                
                distance = ((center_x - cached_center_x) ** 2 + (center_y - cached_center_y) ** 2) ** 0.5
                position_match = distance < self.position_tolerance * 1.5
                
                # Calculate embedding similarity if available
                embedding_similarity = 0.0
                if encoding is not None and entry.get('encoding') is not None:
                    embedding_similarity = self._compute_embedding_similarity(encoding, entry['encoding'])
                
                # Score prioritizes embedding match over position
                if embedding_similarity > self.embedding_threshold:
                    # Same face based on embedding - strong match
                    score = embedding_similarity + (0.2 if position_match else 0)
                    if score > best_score:
                        best_score = score
                        best_match = entry
                elif position_match and distance < self.position_tolerance:
                    # Position-only match (fallback when no embedding provided)
                    score = 0.5 - (distance / self.position_tolerance) * 0.3
                    if encoding is None and score > best_score:
                        best_score = score
                        best_match = entry
            
            # Remove expired entries
            for key in expired_keys:
                del self.cache[key]
        
        return best_match
    
    def get(self, location: Tuple[int, int, int, int], 
            encoding: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
        """
        Retrieve cached recognition result by location and/or embedding.
        
        Args:
            location: Face bounding box (top, right, bottom, left)
            encoding: Face embedding vector for similarity matching
            
        Returns:
            Cached result dict with name, confidence, etc., or None if not found
        """
        return self._find_nearby_cache(location, encoding)
    
    def get_by_embedding(self, encoding: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Find cached face purely by embedding similarity.
        
        Used when position matching isn't reliable (e.g., face has moved significantly).
        """
        if encoding is None:
            return None
        
        now = time.time()
        best_match = None
        best_similarity = self.embedding_threshold
        
        with self.lock:
            for key, entry in self.cache.items():
                # Skip expired entries
                if now - entry['timestamp'] > self.ttl:
                    continue
                if entry.get('encoding') is None:
                    continue
                
                similarity = self._compute_embedding_similarity(encoding, entry['encoding'])
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = entry
        
        return best_match
    
    def put(self, location: Tuple[int, int, int, int], name: str, confidence: float, 
            encoding: Optional[np.ndarray] = None) -> None:
        """
        Store a recognition result in the cache.
        
        Args:
            location: Face bounding box for position-based lookup
            name: Recognized person's name
            confidence: Recognition confidence score
            encoding: Face embedding for similarity-based matching
        """
        key = self._get_position_key(location)
        with self.lock:
            # Enforce cache size limit
            if len(self.cache) >= self.max_entries:
                self._evict_oldest_entries()
            
            # Store embedding as float32 to reduce memory usage
            stored_encoding = None
            if encoding is not None:
                stored_encoding = np.asarray(encoding, dtype=np.float32)
            
            self.cache[key] = {
                'name': name,
                'confidence': confidence,
                'location': location,
                'timestamp': time.time(),
                'encoding': stored_encoding
            }
    
    def _evict_oldest_entries(self):
        """
        Remove oldest 25% of entries when cache is full.
        Must be called while holding the lock.
        """
        if not self.cache:
            return
        entries = sorted(self.cache.items(), key=lambda x: x[1]['timestamp'])
        num_to_remove = max(1, len(entries) // 4)
        for key, _ in entries[:num_to_remove]:
            del self.cache[key]
    
    def update_location(self, old_location: Tuple[int, int, int, int], 
                        new_location: Tuple[int, int, int, int]) -> None:
        """
        Update cache when a tracked face moves to a new position.
        Refreshes the TTL timestamp to keep the entry alive.
        """
        old_key = self._get_position_key(old_location)
        new_key = self._get_position_key(new_location)
        with self.lock:
            if old_key in self.cache:
                entry = self.cache[old_key]
                entry['location'] = new_location
                entry['timestamp'] = time.time()
                if old_key != new_key:
                    self.cache[new_key] = entry
                    del self.cache[old_key]
    
    def clear(self) -> None:
        """Remove all entries from the cache."""
        with self.lock:
            self.cache.clear()
    
    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from cache.
        
        Returns:
            Number of entries removed
        """
        now = time.time()
        with self.lock:
            expired_keys = [k for k, v in self.cache.items() if now - v['timestamp'] > self.ttl]
            for key in expired_keys:
                del self.cache[key]
            return len(expired_keys)
    
    def get_all_active(self) -> List[Dict[str, Any]]:
        """Get all non-expired cached entries for display purposes."""
        now = time.time()
        with self.lock:
            return [entry.copy() for entry in self.cache.values() 
                    if now - entry['timestamp'] <= self.ttl]


# ==================== DOOR CONTROLLER ====================
class DoorController:
    """
    Controls the physical door lock mechanism.
    
    Supports GPIO-based relay control for Raspberry Pi deployments,
    with automatic fallback to simulation mode on non-Pi systems.
    """
    
    def __init__(self):
        """Initialize door controller and set up GPIO if available."""
        self.is_unlocked = False
        self.unlock_thread = None
        
        # Configure GPIO for relay control on Raspberry Pi
        if USE_GPIO:
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(Config.DOOR_RELAY_PIN, GPIO.OUT)
            GPIO.output(Config.DOOR_RELAY_PIN, GPIO.LOW)  # Start locked
    
    def unlock(self, duration=None):
        """
        Unlock the door for a specified duration.
        
        Args:
            duration: Seconds to keep door unlocked (uses config default if None)
        """
        if duration is None:
            duration = Config.DOOR_UNLOCK_DURATION
        
        # Prevent overlapping unlock operations
        if self.unlock_thread and self.unlock_thread.is_alive():
            return
        
        self.unlock_thread = threading.Thread(target=self._unlock_sequence, args=(duration,))
        self.unlock_thread.start()
    
    def _unlock_sequence(self, duration):
        """
        Execute the timed unlock sequence.
        Runs in a separate thread to avoid blocking the main loop.
        """
        self.is_unlocked = True
        
        if USE_GPIO:
            GPIO.output(Config.DOOR_RELAY_PIN, GPIO.HIGH)  # Activate relay
        
        logger.info(f"Door unlocked for {duration} seconds")
        time.sleep(duration)
        
        if USE_GPIO:
            GPIO.output(Config.DOOR_RELAY_PIN, GPIO.LOW)  # Deactivate relay
        
        self.is_unlocked = False
        logger.info("Door locked")
    
    def cleanup(self):
        """Release GPIO resources on application exit."""
        if USE_GPIO:
            GPIO.cleanup()


# ==================== ACCESS LOG ====================
class AccessLog:
    """
    Manages access event logging with persistent storage.
    
    Stores access attempts (granted/denied) with timestamps.
    Uses memory-efficient storage keeping only recent entries in RAM
    while maintaining full history on disk.
    """
    
    def __init__(self, log_path=None):
        """
        Initialize access log.
        
        Args:
            log_path: Path to JSON log file (uses config default if None)
        """
        self.log_path = log_path or Config.ACCESS_LOG_PATH
        self.max_memory_entries = getattr(Config, 'ACCESS_LOG_MAX_MEMORY_ENTRIES', 1000)
        self.entries = []  # Recent entries kept in memory
        self._total_entries = 0  # Total count including on-disk entries
        self.load()
    
    def load(self):
        """
        Load log entries from disk.
        Only keeps the most recent entries in memory to limit RAM usage.
        """
        if os.path.exists(self.log_path):
            try:
                with open(self.log_path, 'r') as f:
                    all_entries = json.load(f)
                    self._total_entries = len(all_entries)
                    # Keep only recent entries in memory
                    self.entries = all_entries[-self.max_memory_entries:]
            except (json.JSONDecodeError, IOError):
                self.entries = []
                self._total_entries = 0
        else:
            self.entries = []
            self._total_entries = 0
    
    def save(self):
        """
        Persist log entries to disk.
        Merges in-memory entries with existing disk entries to prevent duplicates.
        """
        try:
            # Load existing entries from disk
            if os.path.exists(self.log_path):
                with open(self.log_path, 'r') as f:
                    try:
                        all_entries = json.load(f)
                    except (json.JSONDecodeError, IOError):
                        all_entries = []
            else:
                all_entries = []
            
            # Merge: deduplicate by timestamp to avoid duplicate entries
            existing_timestamps = {e['timestamp'] for e in all_entries}
            new_entries = [e for e in self.entries if e['timestamp'] not in existing_timestamps]
            all_entries.extend(new_entries)
            
            with open(self.log_path, 'w') as f:
                json.dump(all_entries, f, indent=2)
            
            self._total_entries = len(all_entries)
        except IOError as e:
            logger.error(f"Failed to save access log: {e}")
    
    def add_entry(self, name, access_granted, confidence=0.0):
        """
        Record a new access event.
        
        Args:
            name: Name of person attempting access
            access_granted: Whether access was allowed
            confidence: Recognition confidence score
            
        Returns:
            The created log entry dictionary
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "name": name,
            "access_granted": access_granted,
            "confidence": round(confidence, 3)
        }
        self.entries.append(entry)
        self._total_entries += 1
        
        # Limit memory usage by trimming old entries
        if len(self.entries) > self.max_memory_entries:
            self.entries = self.entries[-self.max_memory_entries:]
        
        self.save()
        return entry
    
    def get_recent(self, count=50):
        """Get the most recent N log entries in reverse chronological order."""
        return list(reversed(self.entries[-count:]))
    
    def get_filtered(self, date_from=None, date_to=None, name_filter=None, count=100):
        """
        Query log entries with optional filters.
        
        Args:
            date_from: Earliest date to include (inclusive)
            date_to: Latest date to include (inclusive)
            name_filter: Partial name match (case-insensitive)
            count: Maximum entries to return
            
        Returns:
            List of matching entries in reverse chronological order
        """
        filtered = []
        for entry in reversed(self.entries):
            entry_date = datetime.fromisoformat(entry['timestamp']).date()
            
            # Apply date range filter
            if date_from and entry_date < date_from:
                continue
            if date_to and entry_date > date_to:
                continue
            
            # Apply name filter (case-insensitive substring match)
            if name_filter and name_filter.lower() not in entry['name'].lower():
                continue
            
            filtered.append(entry)
            if len(filtered) >= count:
                break
        
        return filtered
    
    def get_unique_names(self):
        """Get sorted list of all unique names in the access log."""
        names = set()
        for entry in self.entries:
            names.add(entry['name'])
        return sorted(list(names))
    
    def clear(self):
        """Delete all log entries from memory and disk."""
        self.entries = []
        self._total_entries = 0
        # Overwrite the JSON file with empty array (don't use save() which merges)
        try:
            with open(self.log_path, 'w') as f:
                json.dump([], f)
        except IOError as e:
            logger.error(f"Failed to clear access log file: {e}")


# ==================== CAMERA MANAGER ====================
class CameraManager:
    """
    Manages camera hardware with background frame capture.
    
    Runs a dedicated thread for continuous frame capture to ensure
    the main thread always has the latest frame available without blocking.
    Supports both USB webcams and Raspberry Pi camera module.
    """
    
    def __init__(self, use_picamera=False, resolution=(640, 480)):
        """
        Initialize camera manager.
        
        Args:
            use_picamera: Use Raspberry Pi camera instead of USB webcam
            resolution: Capture resolution as (width, height)
        """
        self.use_picamera = use_picamera
        self.resolution = resolution
        self.camera = None
        self.is_running = False
        
        # Thread-safe frame buffer
        self.current_frame = None
        self.frame_lock = threading.Lock()
        self.capture_thread = None
    
    def start(self):
        """
        Initialize camera hardware and start background capture.
        Automatically probes for available camera devices.
        """
        if self.use_picamera:
            # Raspberry Pi camera module
            self.camera = Picamera2()
            self.camera.configure(self.camera.create_preview_configuration(
                main={"format": 'XRGB8888', "size": self.resolution}, buffer_count=2
            ))
            self.camera.start()
        else:
            # USB webcam - probe multiple indices for flexibility
            camera_indices = [0, 1, 2]
            self.camera = None
            
            for idx in camera_indices:
                cap = cv2.VideoCapture(idx)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        self.camera = cap
                        logger.info(f"Camera connected to video device {idx}")
                        break
                    else:
                        cap.release()
                else:
                    cap.release()
            
            if self.camera is None:
                raise RuntimeError("Could not connect to any camera. Tried indices: 0, 1, 2")
            
            # Configure camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
        
        self.is_running = True
        
        # Start background capture thread
        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.capture_thread.start()
        
        # Allow camera to warm up
        time.sleep(0.3)
    
    def _capture_loop(self):
        """
        Continuous frame capture running in background thread.
        Ensures latest frame is always available for the main loop.
        """
        while self.is_running:
            try:
                if self.use_picamera:
                    frame = self.camera.capture_array()
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                else:
                    ret, frame = self.camera.read()
                    if not ret:
                        continue
                
                # Thread-safe frame update
                with self.frame_lock:
                    self.current_frame = frame
                    
            except Exception as e:
                logger.warning(f"Camera capture error: {e}")
                time.sleep(0.1)
    
    def capture_frame(self):
        """
        Get the most recent captured frame.
        Returns immediately with latest frame (non-blocking).
        """
        if not self.is_running:
            return None
        
        with self.frame_lock:
            if self.current_frame is not None:
                return self.current_frame.copy()
        return None
    
    def stop(self):
        """Stop capture thread and release camera hardware."""
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
    """
    Core face recognition engine using InsightFace (buffalo_s model).
    
    Features:
    - Background threaded recognition to prevent UI blocking
    - Face stability detection to avoid recognizing moving faces
    - Embedding caching to prevent redundant recognition
    - Vectorized cosine similarity for fast matching
    - Support for CUDA and CPU execution providers
    """
    
    def __init__(self, dataset_path=None, encodings_path=None):
        """
        Initialize the face recognition system.
        
        Args:
            dataset_path: Directory containing person face images
            encodings_path: Path to pickle file with face encodings
        """
        self.dataset_path = dataset_path or Config.DATASET_PATH
        self.encodings_path = encodings_path or Config.ENCODINGS_PATH
        self.known_encodings = []
        self.known_names = []
        self.known_encodings_normalized = None  # Pre-computed for fast matching
        self.cv_scaler = Config.DETECTION_SCALE_FACTOR
        
        # Face stability tracking - wait for face to settle before recognizing
        self.stability_tracker = FaceStabilityTracker()
        
        # Initialize InsightFace model
        try:
            self.face_app = FaceAnalysis(
                name='buffalo_s',  # Lightweight model suitable for edge devices
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info("InsightFace model (buffalo_s) loaded successfully")
        except Exception as e:
            raise RuntimeError(
                f"Could not load InsightFace model: {e}\n"
                f"Please ensure insightface is properly installed with: pip install insightface onnxruntime"
            )
        
        # Optional Haar cascade for fast fallback detection
        try:
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if os.path.exists(cascade_path):
                self.face_cascade = cv2.CascadeClassifier(cascade_path)
            else:
                self.face_cascade = None
        except Exception:
            self.face_cascade = None
        
        # Cache to avoid re-recognizing the same face
        self.face_cache = FaceCache()
        
        # Frame skip counter for performance
        self.frame_count = 0
        
        # ========== THREADED RECOGNITION ==========
        # Background thread prevents recognition from blocking the camera loop
        self._recognition_thread = None
        self._recognition_lock = threading.Lock()
        self._frame_queue = Queue(maxsize=2)  # Limit queue size to prevent memory buildup
        self._last_results = []  # Most recent recognition results
        self._last_results_lock = threading.Lock()
        self._recognition_running = False
        self._stop_recognition = threading.Event()
        
        # Ensure dataset directory exists
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
        
        self.load_encodings()
    
    def load_encodings(self):
        """
        Load face encodings from disk and prepare for fast matching.
        Pre-normalizes all embeddings for vectorized cosine similarity.
        """
        if os.path.exists(self.encodings_path):
            try:
                with open(self.encodings_path, "rb") as f:
                    data = pickle.loads(f.read())
                    
                # Convert to float32 arrays for memory efficiency
                self.known_encodings = [np.array(enc, dtype=np.float32) for enc in data["encodings"]]
                self.known_names = data["names"]
                
                # Pre-normalize for vectorized similarity computation
                self._update_normalized_encodings()
                logger.info(f"Loaded {len(self.known_encodings)} encodings for {len(set(self.known_names))} persons")
                return True
            except Exception as e:
                logger.error(f"Failed to load encodings: {e}")
                return False
        return False
    
    def _update_normalized_encodings(self):
        """
        Pre-compute normalized encoding matrix for fast similarity.
        Normalizing once here avoids per-comparison normalization overhead.
        """
        if len(self.known_encodings) > 0:
            encodings_matrix = np.array(self.known_encodings, dtype=np.float32)
            norms = np.linalg.norm(encodings_matrix, axis=1, keepdims=True)
            self.known_encodings_normalized = (encodings_matrix / norms).astype(np.float32)
        else:
            self.known_encodings_normalized = None
    
    def get_registered_persons(self):
        """
        Get list of all persons with face images in the dataset folder.
        
        Returns:
            List of tuples: (person_name, image_count)
        """
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
        """Get list of unique names that have been trained (have encodings)."""
        return list(set(self.known_names))
    
    def create_person_folder(self, name):
        """Create a directory for storing a person's face images."""
        person_folder = os.path.join(self.dataset_path, name)
        if not os.path.exists(person_folder):
            os.makedirs(person_folder)
        return person_folder
    
    def save_face_image(self, frame, name):
        """
        Save a captured face image to the person's folder.
        
        Args:
            frame: OpenCV image (BGR format)
            name: Person's name for folder organization
            
        Returns:
            Path to the saved image file
        """
        folder = self.create_person_folder(name)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{name}_{timestamp}.jpg"
        filepath = os.path.join(folder, filename)
        cv2.imwrite(filepath, frame)
        return filepath
    
    def train_model(self, progress_callback=None):
        """
        Train face recognition on all images in the dataset.
        
        Scans all person folders, extracts face embeddings from each image,
        and saves the encodings to disk.
        
        Args:
            progress_callback: Optional function(current, total, path) for progress updates
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        from imutils import paths
        
        image_paths = list(paths.list_images(self.dataset_path))
        if not image_paths:
            return False, "No images found in dataset folder"
        
        known_encodings = []
        known_names = []
        
        for i, image_path in enumerate(image_paths):
            if progress_callback:
                progress_callback(i + 1, len(image_paths), image_path)
            
            # Extract person name from folder structure
            name = image_path.split(os.path.sep)[-2]
            
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            # Get face embeddings using InsightFace
            faces = self.face_app.get(image)
            
            for face in faces:
                if face.embedding is not None:
                    known_encodings.append(face.embedding)
                    known_names.append(name)
        
        if not known_encodings:
            return False, "No faces detected in any images"
        
        # Backup existing encodings before overwriting
        if Config.BACKUP_ENABLED:
            BackupManager.create_backup(self.encodings_path, Config.MAX_BACKUPS)
        
        # Save encodings to disk
        data = {"encodings": [enc.tolist() for enc in known_encodings], "names": known_names}
        with open(self.encodings_path, "wb") as f:
            f.write(pickle.dumps(data))
        
        # Update in-memory encodings
        self.known_encodings = [np.array(enc) for enc in known_encodings]
        self.known_names = known_names
        self._update_normalized_encodings()
        
        return True, f"Training complete! {len(known_encodings)} encodings from {len(set(known_names))} persons"
    
    def train_single_person(self, person_name, progress_callback=None):
        """
        Incrementally train a single person without retraining everyone.
        
        Removes any existing encodings for this person first, then adds
        new encodings from their current images.
        
        Args:
            person_name: Name of person to train
            progress_callback: Optional progress update function
            
        Returns:
            Tuple of (success: bool, message: str)
        """
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
            
            faces = self.face_app.get(image)
            
            for face in faces:
                if face.embedding is not None:
                    new_encodings.append(face.embedding)
                    new_names.append(person_name)
        
        if not new_encodings:
            return False, f"No faces detected in images for {person_name}"
        
        # Remove existing encodings for this person (allows re-training)
        indices_to_keep = [i for i, name in enumerate(self.known_names) if name != person_name]
        self.known_encodings = [self.known_encodings[i] for i in indices_to_keep]
        self.known_names = [self.known_names[i] for i in indices_to_keep]
        
        # Add new encodings
        self.known_encodings.extend(new_encodings)
        self.known_names.extend(new_names)
        
        # Create backup before saving
        if Config.BACKUP_ENABLED:
            BackupManager.create_backup(self.encodings_path, Config.MAX_BACKUPS)
        
        # Persist updated encodings to disk
        data = {"encodings": [enc.tolist() if hasattr(enc, 'tolist') else enc for enc in self.known_encodings], "names": self.known_names}
        with open(self.encodings_path, "wb") as f:
            f.write(pickle.dumps(data))
        
        # Refresh normalized matrix for similarity calculations
        self._update_normalized_encodings()
        
        # Clear cache since encodings changed
        self.face_cache.clear()
        
        return True, f"Added {len(new_encodings)} encodings for {person_name}"
    
    def remove_person_from_model(self, person_name):
        """
        Remove all encodings for a person from the trained model.
        
        Args:
            person_name: Name of person to remove
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        if not self.known_names:
            return False, "No trained model exists"
        
        if person_name not in self.known_names:
            return True, f"{person_name} not found in model (already removed or never trained)"
        
        count_before = len(self.known_encodings)
        
        # Filter out encodings for this person
        indices_to_keep = [i for i, name in enumerate(self.known_names) if name != person_name]
        self.known_encodings = [self.known_encodings[i] for i in indices_to_keep]
        self.known_names = [self.known_names[i] for i in indices_to_keep]
        
        count_removed = count_before - len(self.known_encodings)
        
        # Backup before modifying disk file
        if Config.BACKUP_ENABLED:
            BackupManager.create_backup(self.encodings_path, Config.MAX_BACKUPS)
        
        # Update or remove the encodings file
        if self.known_encodings:
            data = {"encodings": [enc.tolist() if hasattr(enc, 'tolist') else enc for enc in self.known_encodings], "names": self.known_names}
            with open(self.encodings_path, "wb") as f:
                f.write(pickle.dumps(data))
            self._update_normalized_encodings()
        else:
            # No encodings left, clean up
            if os.path.exists(self.encodings_path):
                os.remove(self.encodings_path)
            self.known_encodings_normalized = None
        
        self.face_cache.clear()
        
        return True, f"Removed {count_removed} encodings for {person_name}"
    
    def detect_faces_fast(self, frame):
        """
        Fast face detection using Haar cascade.
        Less accurate but faster than InsightFace detection.
        Falls back to InsightFace if Haar cascade unavailable.
        """
        if self.face_cascade is None or self.face_cascade.empty():
            return self.detect_faces_robust(frame)
        
        # Downscale for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=1/self.cv_scaler, fy=1/self.cv_scaler)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        # Convert coordinates and scale back to original resolution
        locations = []
        for (x, y, w, h) in faces:
            top = y * self.cv_scaler
            right = (x + w) * self.cv_scaler
            bottom = (y + h) * self.cv_scaler
            left = x * self.cv_scaler
            locations.append((top, right, bottom, left))
        
        return locations
    
    def detect_faces_robust(self, frame):
        """
        Robust face detection using InsightFace.
        More accurate and handles various face angles well.
        Preferred method for face registration.
        """
        faces = self.face_app.get(frame)
        
        # Convert InsightFace bbox [left, top, right, bottom] to standard format
        face_locations = []
        for face in faces:
            bbox = face.bbox.astype(int)
            left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
            face_locations.append((top, right, bottom, left))
        
        return face_locations
    
    def detect_faces_combined(self, frame):
        """
        Hybrid detection: fast Haar cascade first, then InsightFace fallback.
        Optimizes for speed while maintaining detection reliability.
        """
        faces = self.detect_faces_fast(frame)
        
        # Fall back to robust detection if fast detection finds nothing
        if not faces:
            faces = self.detect_faces_robust(frame)
        
        return faces
    
    # ========== BACKGROUND RECOGNITION THREAD METHODS ==========
    
    def start_recognition_thread(self):
        """Launch the background recognition processing thread."""
        if self._recognition_thread is not None and self._recognition_thread.is_alive():
            return
        
        self._stop_recognition.clear()
        self._recognition_thread = threading.Thread(target=self._recognition_loop, daemon=True)
        self._recognition_thread.start()
        logger.info("Background recognition thread started")
    
    def stop_recognition_thread(self):
        """Gracefully shut down the recognition thread."""
        self._stop_recognition.set()
        
        # Drain queue to unblock the worker thread
        try:
            while not self._frame_queue.empty():
                self._frame_queue.get_nowait()
        except Empty:
            pass
        
        if self._recognition_thread is not None:
            self._recognition_thread.join(timeout=2.0)
            self._recognition_thread = None
        logger.info("Background recognition thread stopped")
    
    def _recognition_loop(self):
        """
        Main loop for background recognition thread.
        Continuously pulls frames from queue and processes them.
        """
        while not self._stop_recognition.is_set():
            try:
                frame = self._frame_queue.get(timeout=0.1)
            except Empty:
                continue
            
            with self._recognition_lock:
                self._recognition_running = True
                try:
                    results = self._process_recognition(frame)
                    with self._last_results_lock:
                        self._last_results = results
                except Exception as e:
                    logger.error(f"Recognition error: {e}")
                finally:
                    self._recognition_running = False
                    # Help GC by clearing reference
                    del frame
    
    def _process_recognition(self, frame):
        """Internal method to perform actual recognition (runs in background thread)"""
        if not self.known_encodings or self.known_encodings_normalized is None:
            return []
        
        results = []
        
        # Run InsightFace detection and embedding extraction
        faces = self.face_app.get(frame)
        
        for face_idx, face in enumerate(faces):
            if face.embedding is None:
                continue
            
            face_encoding = face.embedding
            bbox = face.bbox.astype(int)
            left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
            location = (top, right, bottom, left)
            
            # Check face stability before attempting recognition
            is_stable = self.stability_tracker.update_and_check_stability(face_idx, location)
            
            # Try cache lookup by embedding similarity first
            cached = self.face_cache.get_by_embedding(face_encoding)
            if cached:
                # Update cache position since face may have moved
                self.face_cache.update_location(cached['location'], location)
                results.append({
                    'name': cached['name'],
                    'confidence': cached['confidence'],
                    'location': location,
                    'from_cache': True,
                    'is_stable': is_stable
                })
                continue
            
            # Fallback: check cache by position
            cached = self.face_cache.get(location, face_encoding)
            if cached:
                results.append({
                    'name': cached['name'],
                    'confidence': cached['confidence'],
                    'location': location,
                    'from_cache': True,
                    'is_stable': is_stable
                })
                continue
            
            # Defer recognition for moving faces
            if not is_stable:
                results.append({
                    'name': 'Scanning...',
                    'confidence': 0.0,
                    'location': location,
                    'from_cache': False,
                    'is_stable': False
                })
                continue
            
            # Perform actual recognition using vectorized cosine similarity
            name = "Unknown"
            confidence = 0.0
            
            # Normalize current embedding
            face_norm = face_encoding / np.linalg.norm(face_encoding)
            
            # Compute similarities against all known faces in one operation
            similarities = np.dot(self.known_encodings_normalized, face_norm)
            
            best_match_index = np.argmax(similarities)
            best_similarity = similarities[best_match_index]
            
            if best_similarity > Config.RECOGNITION_THRESHOLD:
                name = self.known_names[best_match_index]
                confidence = float(best_similarity)
            
            # Cache result for future frames
            self.face_cache.put(location, name, confidence, face_encoding)
            
            results.append({
                'name': name,
                'confidence': confidence,
                'location': location,
                'from_cache': False,
                'is_stable': True
            })
        
        return results
    
    def recognize_faces(self, frame, force_recognition=False):
        """
        Non-blocking face recognition using background thread.
        
        Submits frame for processing and immediately returns cached/previous results.
        This prevents the camera loop from being blocked by slow recognition.
        
        Args:
            frame: OpenCV BGR image
            force_recognition: Skip frame interval check
            
        Returns:
            Tuple of (frame, list of recognition results)
        """
        self.frame_count += 1
        
        if not self.known_encodings or self.known_encodings_normalized is None:
            return frame, []
        
        # Implement frame skipping for performance
        if not force_recognition and self.frame_count % Config.RECOGNITION_INTERVAL_FRAMES != 0:
            with self._last_results_lock:
                return frame, self._last_results.copy() if self._last_results else []
        
        # Submit frame to background thread (non-blocking)
        try:
            # Discard old frames to ensure we process the most recent
            while not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except Empty:
                    break
            self._frame_queue.put_nowait(frame.copy())
        except:
            pass  # Queue full - drop this frame
        
        # Return immediately with cached results
        with self._last_results_lock:
            return frame, self._last_results.copy() if self._last_results else []
    
    def recognize_faces_sync(self, frame, force_recognition=False):
        """
        Synchronous face recognition (blocks until complete).
        Use for single-frame recognition scenarios.
        """
        if not self.known_encodings or self.known_encodings_normalized is None:
            return frame, []
        
        return frame, self._process_recognition(frame)
    
    def get_last_results(self):
        """Retrieve the most recent recognition results."""
        with self._last_results_lock:
            return self._last_results.copy() if self._last_results else []
    
    def is_recognition_busy(self):
        """Check if recognition thread is currently processing a frame."""
        return self._recognition_running
    
    def clear_cache(self):
        """Reset face cache, stability tracker, and pending frame queue."""
        self.face_cache.clear()
        self.stability_tracker.clear()
        with self._last_results_lock:
            self._last_results = []
        try:
            while not self._frame_queue.empty():
                self._frame_queue.get_nowait()
        except Empty:
            pass


# ==================== ON-SCREEN KEYBOARD ====================
class OnScreenKeyboard:
    """
    Touch-friendly virtual keyboard for kiosk text entry.
    
    Designed for touchscreen environments where physical keyboard
    is not available. Supports shift for uppercase and special characters.
    """
    
    _active_keyboard = None  # Singleton to prevent duplicate keyboards
    
    def __init__(self, container, entry_widget, root_window):
        """
        Create a virtual keyboard attached to an entry widget.
        
        Args:
            container: Parent frame to embed keyboard into
            entry_widget: Text entry field to receive keystrokes
            root_window: Main application window for event binding
        """
        # Close any existing keyboard to prevent duplicates
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
        self.all_buttons = []
        
        self._create_keyboard()
        
        # Auto-close on Enter key
        self.entry.bind('<Return>', lambda e: self.close())
        
        OnScreenKeyboard._active_keyboard = self
    
    def _is_keyboard_widget(self, widget):
        """Check if a widget belongs to this keyboard."""
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
        """Build the keyboard layout and embed it in the container."""
        self.keyboard_frame = tk.Frame(self.container, bg=Config.COLOR_BORDER)
        self.keyboard_frame.pack(side=tk.BOTTOM, fill=tk.X)
        
        main_frame = tk.Frame(self.keyboard_frame, bg=Config.COLOR_BG)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)
        
        # Keyboard layout - QWERTY with special keys
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
                # Style special keys differently
                bg_color = Config.COLOR_CARD
                fg_color = Config.COLOR_TEXT
                if key == '✓':
                    bg_color = Config.COLOR_GRANTED
                    fg_color = "#FFFFFF"
                elif key == '✕':
                    bg_color = Config.COLOR_DENIED
                    fg_color = "#FFFFFF"
                
                # Space bar gets more width
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
        """Handle virtual key press event."""
        if key == '✓':
            self.close()
        elif key == '✕':
            self.entry.delete(0, tk.END)
            self.close()
        elif key == '⌫':
            # Backspace - remove last character
            current = self.entry.get()
            self.entry.delete(0, tk.END)
            self.entry.insert(0, current[:-1])
        elif key == '⇧':
            # Toggle shift mode
            self.shift_on = not self.shift_on
            self.shift_btn.config(bg=Config.COLOR_SCANNING if self.shift_on else Config.COLOR_CARD,
                                  fg="#FFFFFF" if self.shift_on else Config.COLOR_TEXT)
        elif key == ' ':
            self.entry.insert(tk.END, ' ')
        else:
            # Regular character - apply shift if active
            char = key.upper() if self.shift_on else key
            self.entry.insert(tk.END, char)
            if self.shift_on:
                self.shift_on = False
                self.shift_btn.config(bg=Config.COLOR_CARD, fg=Config.COLOR_TEXT)
        
        self.entry.focus_set()
    
    def close(self):
        """Destroy keyboard and clean up resources."""
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
    """Convenience function to display the on-screen keyboard."""
    if root_window is None:
        root_window = container
    OnScreenKeyboard(container, entry_widget, root_window)


# ==================== MAIN KIOSK APPLICATION ====================
class DoorEntryKiosk:
    """
    Main application class for the door entry kiosk system.
    
    Manages the complete kiosk interface including:
    - Live camera feed with face detection overlay
    - Face recognition and access control
    - Admin panel for user management and training
    - Registration workflow for new users
    - Access logging and statistics
    """
    
    def __init__(self, root):
        """
        Initialize the kiosk application.
        
        Args:
            root: Tkinter root window
        """
        self.root = root
        self.root.title(Config.WINDOW_TITLE)
        
        # Configure window mode
        if Config.FULLSCREEN:
            self.root.attributes('-fullscreen', True)
            self.root.bind('<Escape>', lambda e: self.toggle_fullscreen())
        else:
            self.root.geometry("480x800")
            self.root.resizable(False, False)
        
        self.root.configure(bg=Config.COLOR_BG)
        
        # Initialize core system components
        self.camera = CameraManager(use_picamera=USE_PICAMERA, resolution=Config.CAMERA_RESOLUTION)
        self.face_system = FaceRecognitionSystem()
        self.door_controller = DoorController()
        self.access_log = AccessLog()
        
        # ========== Application State ==========
        self.is_running = True
        self.is_scanning = True
        self.camera_thread = None
        self.current_status = "scanning"  # States: scanning, granted, denied
        self.status_message = ""
        self.last_access = {}  # Cooldown tracking per person
        self.admin_mode = False
        
        # ========== Registration State ==========
        self.registration_mode = False
        self.registration_name = ""
        self.captured_count = 0
        self.current_frame = None
        
        # Face ID style auto-capture settings
        self.auto_capture_mode = False
        self.auto_capture_target = 100
        self.auto_capture_interval = 0.2  # Seconds between captures
        self.last_auto_capture = 0
        
        # Zone-based capture for multi-angle coverage
        self.zone_captures = {'center': 0, 'left': 0, 'right': 0, 'up': 0, 'down': 0}
        self.current_zone = 'center'
        self.zone_targets = {'center': 30, 'left': 18, 'right': 18, 'up': 17, 'down': 17}
        
        # ========== Training State ==========
        self.is_training = False
        self.reg_process_locked = False  # Prevents concurrent InsightFace operations
        
        # ========== Performance Tracking ==========
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0.0
        self.faces_detected = 0
        self.cache_hits = 0
        self.cache_misses = 0
        
        # ========== User Management State ==========
        self.person_map = {}  # Maps listbox indices to person names
        
        # ========== Memory Management ==========
        self.last_cache_cleanup = time.time()
        self.cache_cleanup_interval = 60  # Cleanup every 60 seconds
        
        # UI state tracking
        self._status_reset_id = None  # For cancelling pending status resets
        self._toast_id = None  # For toast notification timing
        self._pending_toasts = []  # Queue of toast notifications
        
        # Build the user interface
        self.create_kiosk_interface()
        
        # Initialize camera system
        self.start_camera()
        
        # Handle window close properly
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        
        # Keyboard shortcuts
        self.root.bind('<F1>', lambda e: self.show_admin_login())
        self.root.bind('<F11>', lambda e: self.toggle_fullscreen())
    
    def toggle_fullscreen(self):
        """Toggle between fullscreen and windowed mode."""
        is_fullscreen = self.root.attributes('-fullscreen')
        self.root.attributes('-fullscreen', not is_fullscreen)
    
    def create_kiosk_interface(self):
        """
        Build the main kiosk interface with Apple-inspired design.
        Creates the camera view, status display, and navigation elements.
        """
        self.root.configure(bg=Config.COLOR_BG)
        
        # Main container frame
        self.main_frame = tk.Frame(self.root, bg=Config.COLOR_BG)
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # ===== TOP STATUS BAR =====
        top_bar = tk.Frame(self.main_frame, bg=Config.COLOR_BG, height=40)
        top_bar.pack(fill=tk.X, padx=15, pady=(10, 0))
        top_bar.pack_propagate(False)
        
        # Current time display
        self.time_label = tk.Label(
            top_bar,
            text="",
            font=(Config.FONT_FAMILY, 12),
            fg=Config.COLOR_TEXT_SECONDARY,
            bg=Config.COLOR_BG
        )
        self.time_label.pack(side=tk.LEFT, pady=8)
        self.update_time()
        
        # Live camera indicator
        self.fps_indicator = tk.Label(
            top_bar,
            text="● LIVE",
            font=(Config.FONT_FAMILY, 9, "bold"),
            fg=Config.COLOR_GRANTED,
            bg=Config.COLOR_BG
        )
        self.fps_indicator.pack(side=tk.RIGHT, pady=8)
        
        # ===== CENTER CONTENT AREA =====
        center_frame = tk.Frame(self.main_frame, bg=Config.COLOR_BG)
        center_frame.pack(fill=tk.BOTH, expand=True, padx=15, pady=10)
        
        # Camera feed container
        camera_wrapper = tk.Frame(center_frame, bg=Config.COLOR_BG)
        camera_wrapper.pack(expand=True)
        
        # Video display with card styling
        self.video_container = tk.Frame(
            camera_wrapper, 
            bg=Config.COLOR_CARD,
            highlightbackground=Config.COLOR_BORDER,
            highlightthickness=1
        )
        self.video_container.pack()
        self.video_container.pack_propagate(False)
        
        # Set fixed video dimensions
        video_width = 440
        video_height = 330
        self.video_container.config(width=video_width, height=video_height)
        
        self.video_label = tk.Label(self.video_container, bg="#000000")
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # ===== STATUS DISPLAY =====
        status_frame = tk.Frame(camera_wrapper, bg=Config.COLOR_BG)
        status_frame.pack(pady=(15, 0))
        
        # Status card with pill-style appearance
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
        
        # Status icon (color-coded)
        self.status_icon_label = tk.Label(
            status_inner,
            text="◉",
            font=(Config.FONT_FAMILY, 24),
            fg=Config.COLOR_SCANNING,
            bg=Config.COLOR_CARD
        )
        self.status_icon_label.pack(side=tk.LEFT, padx=(0, 10))
        
        # Status text container
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
        
        self.status_text_frame = status_text_frame
        
        # ===== BOTTOM NAVIGATION BAR =====
        bottom_bar = tk.Frame(self.main_frame, bg=Config.COLOR_BG, height=50)
        bottom_bar.pack(fill=tk.X, side=tk.BOTTOM, padx=15, pady=(0, 10))
        bottom_bar.pack_propagate(False)
        
        # Settings button
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
        
        # Building/company branding
        self.title_label = tk.Label(
            bottom_bar,
            text="EDUWEL",
            font=(Config.FONT_FAMILY, 10, "bold"),
            fg=Config.COLOR_TEXT_TERTIARY,
            bg=Config.COLOR_BG,
            anchor="center"
        )
        self.title_label.pack(side=tk.LEFT, expand=True, pady=10)
        
        # Registered user count
        self.info_label = tk.Label(
            bottom_bar,
            text=f"{len(self.face_system.get_trained_persons())} Users",
            font=(Config.FONT_FAMILY, 9),
            fg=Config.COLOR_TEXT_TERTIARY,
            bg=Config.COLOR_BG
        )
        self.info_label.pack(side=tk.RIGHT, pady=10)
        
        # Hidden listbox for access log (visible in admin panel only)
        self.log_listbox = tk.Listbox(self.main_frame)
        self.log_listbox.pack_forget()
        
        # Load recent log entries
        self.update_log_display()
    
    def update_time(self):
        """Update the clock display in the top bar."""
        current_time = datetime.now().strftime("%H:%M:%S")
        current_date = datetime.now().strftime("%a, %b %d %Y")
        self.time_label.config(text=f"{current_time}  ·  {current_date}")
        self.root.after(1000, self.update_time)
    
    def update_log_display(self):
        """Refresh the access log listbox with recent entries."""
        self.log_listbox.delete(0, tk.END)
        entries = self.access_log.get_recent(8)
        
        for entry in entries:
            # Format timestamps as relative or absolute
            entry_time = datetime.fromisoformat(entry['timestamp'])
            now = datetime.now()
            diff = now - entry_time
            
            if diff.total_seconds() < 60:
                time_str = "Just now"
            elif diff.total_seconds() < 3600:
                mins = int(diff.total_seconds() / 60)
                time_str = f"{mins}m ago"
            elif entry_time.date() == now.date():
                time_str = entry_time.strftime("%H:%M")
            else:
                time_str = entry_time.strftime("%b %d")
            
            status_icon = "●" if entry['access_granted'] else "○"
            name = entry['name'][:12] + "..." if len(entry['name']) > 12 else entry['name']
            self.log_listbox.insert(tk.END, f"  {status_icon}  {time_str:>8}   {name}")
    
    def set_status(self, status, name="", confidence=0.0):
        """
        Update the status display with appropriate styling.
        
        Args:
            status: One of "scanning", "active_scanning", "processing", "granted", "denied"
            name: Person name for granted status
            confidence: Recognition confidence for display
        """
        self.current_status = status
        
        # Cancel any pending status reset
        if hasattr(self, '_status_reset_id') and self._status_reset_id:
            self.root.after_cancel(self._status_reset_id)
            self._status_reset_id = None
        
        def update_bg(bg_color, icon_fg, text_fg, detail_fg):
            """Helper to update all widget colors consistently."""
            self.status_card.config(bg=bg_color, highlightbackground=bg_color)
            self.status_frame.config(bg=bg_color)
            self.status_text_frame.config(bg=bg_color)
            self.status_icon_label.config(bg=bg_color, fg=icon_fg)
            self.status_text_label.config(bg=bg_color, fg=text_fg)
            self.status_detail_label.config(bg=bg_color, fg=detail_fg)
        
        if status == "granted":
            display_name = name if len(name) <= 20 else name[:18] + "..."
            self.status_icon_label.config(text="✓")
            self.status_text_label.config(text=f"Welcome, {display_name}")
            self.status_detail_label.config(text=f"Access granted • {confidence:.0%} match")
            update_bg(Config.COLOR_GRANTED, "#FFFFFF", "#FFFFFF", "#E8F5E9")
            # Auto-reset to scanning after display duration
            self._status_reset_id = self.root.after(
                Config.STATUS_DISPLAY_DURATION, 
                self._set_status_scanning
            )
            
        elif status == "denied":
            self.status_icon_label.config(text="✕")
            self.status_text_label.config(text="Not Recognized")
            self.status_detail_label.config(text="Please register or try again")
            update_bg(Config.COLOR_DENIED, "#FFFFFF", "#FFFFFF", "#FFEBEE")
            self._status_reset_id = self.root.after(
                Config.STATUS_DISPLAY_DURATION, 
                self._set_status_scanning
            )
            
        elif status == "active_scanning":
            self.status_icon_label.config(text="◎")
            self.status_text_label.config(text="Scanning...")
            self.status_detail_label.config(text="Hold still for recognition")
            update_bg(Config.COLOR_CARD, Config.COLOR_SCANNING, Config.COLOR_TEXT, Config.COLOR_TEXT_SECONDARY)
            self.status_card.config(highlightbackground=Config.COLOR_SCANNING)
            
        elif status == "processing":
            # Face detected but still settling
            self.status_icon_label.config(text="⏳")
            self.status_text_label.config(text="Processing...")
            self.status_detail_label.config(text="Please wait")
            update_bg(Config.COLOR_CARD, Config.COLOR_WARNING, Config.COLOR_TEXT, Config.COLOR_TEXT_SECONDARY)
            self.status_card.config(highlightbackground=Config.COLOR_WARNING)
            
        else:  # Default scanning state
            self.status_icon_label.config(text="◉")
            self.status_text_label.config(text="Ready to Scan")
            self.status_detail_label.config(text="Look at the camera")
            update_bg(Config.COLOR_CARD, Config.COLOR_SCANNING, Config.COLOR_TEXT, Config.COLOR_TEXT_SECONDARY)
            self.status_card.config(highlightbackground=Config.COLOR_BORDER)
    
    def _set_status_active_scanning(self):
        """Helper method to set active scanning status."""
        self.set_status("active_scanning")
    
    def _set_status_scanning(self):
        """Helper method to reset to idle scanning status."""
        self.set_status("scanning")
    
    def start_camera(self):
        """Initialize camera and start the main processing loop."""
        self.face_system.start_recognition_thread()
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()
    
    def camera_loop(self):
        """
        Main camera processing loop running in background thread.
        Handles frame capture, face recognition, and access decisions.
        """
        try:
            self.camera.start()
            frame_time = time.time()
            
            while self.is_running:
                loop_start = time.time()
                
                # Grab latest frame from camera
                frame = self.camera.capture_frame()
                if frame is None:
                    continue
                
                display_frame = frame.copy()
                
                if self.is_scanning and not self.registration_mode and not self.admin_mode:
                    # Skip recognition during training to avoid conflicts
                    if self.is_training:
                        cv2.putText(display_frame, "Training in progress...", (50, 50),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2)
                    else:
                        # Run face recognition
                        _, results = self.face_system.recognize_faces(frame)
                        self.faces_detected = len(results)
                        
                        # Check face stability status
                        any_unstable = any(not r.get('is_stable', True) for r in results)
                        any_stable = any(r.get('is_stable', True) for r in results)
                        
                        # Update UI status based on face detection
                        if len(results) > 0 and self.current_status not in ("granted", "denied"):
                            if any_unstable and not any_stable:
                                self.root.after(0, lambda: self.set_status("processing"))
                            else:
                                self.root.after(0, self._set_status_active_scanning)
                        elif len(results) == 0 and self.current_status in ("active_scanning", "processing"):
                            self.root.after(0, self._set_status_scanning)
                        
                        # Process each recognized face
                        for result in results:
                            name = result['name']
                            confidence = result['confidence']
                            from_cache = result.get('from_cache', False)
                            is_stable = result.get('is_stable', True)
                            
                            # Skip unstable faces
                            if not is_stable or name == 'Scanning...':
                                continue
                            
                            # Handle recognized faces
                            if name != "Unknown" and confidence >= Config.RECOGNITION_THRESHOLD:
                                now = time.time()
                                # Apply cooldown to prevent duplicate logs
                                if name not in self.last_access or (now - self.last_access[name]) > Config.COOLDOWN_SECONDS:
                                    self.last_access[name] = now
                                    self.root.after(0, lambda n=name, c=confidence: self.grant_access(n, c))
                            else:
                                # Log unknown faces (with cooldown)
                                if name == "Unknown" and self.current_status in ("scanning", "active_scanning") and not from_cache:
                                    now = time.time()
                                    if "Unknown" not in self.last_access or (now - self.last_access["Unknown"]) > Config.COOLDOWN_SECONDS:
                                        self.last_access["Unknown"] = now
                                        self.root.after(0, self.deny_access)
                
                elif self.registration_mode:
                    # Registration mode with zone-based capture
                    frame_height, frame_width = display_frame.shape[:2]
                    
                    if self.auto_capture_mode:
                        # Active face capture with positioning overlay
                        faces = self.face_system.detect_faces_combined(frame)
                        frame_center_x = frame_width // 2
                        frame_center_y = frame_height // 2
                        
                        if len(faces) == 1:
                            top, right, bottom, left = faces[0]
                            
                            # Calculate face position
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
                        # Idle/manual capture mode - just show simple face detection
                        faces = self.face_system.detect_faces_combined(frame)
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
                        logger.debug(f"Cache cleanup: removed {expired_count} expired entries")
                    # Also clean up old last_access entries to prevent memory leak
                    self._cleanup_old_access_entries(now)
                    # Run garbage collection periodically
                    gc.collect()
                    self.last_cache_cleanup = now
                
                # Store current frame reference (avoid unnecessary copy when possible)
                self.current_frame = frame
                
                # Update display
                self.root.after(0, lambda f=display_frame: self.display_frame(f))
                
                # Adaptive frame rate - aim for ~30 FPS
                loop_time = time.time() - loop_start
                sleep_time = max(0.001, 0.033 - loop_time)
                time.sleep(sleep_time)
            
        except Exception as e:
            logger.error(f"Camera error: {e}")
            import traceback
            logger.debug(traceback.format_exc())
            # Show error on UI
            self.root.after(0, lambda: self.show_toast("Camera error - restarting...", "error"))
            # Attempt recovery
            try:
                time.sleep(2)  # Wait before retry
                if self.is_running:
                    logger.info("Attempting camera recovery...")
                    self.camera.stop()
                    time.sleep(1)
                    self.camera.start()
                    logger.info("Camera recovered successfully")
            except Exception as recovery_error:
                logger.error(f"Camera recovery failed: {recovery_error}")
        finally:
            try:
                self.camera.stop()
            except Exception:
                pass
    
    def display_frame(self, frame):
        """
        Display a frame on the video label.
        Also updates admin preview if admin panel is open.
        """
        if USE_PICAMERA:
            frame_rgb = frame
        else:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Ensure consistent display size
        frame_h, frame_w = frame_rgb.shape[:2]
        if frame_w != 440 or frame_h != 330:
            frame_rgb = cv2.resize(frame_rgb, (440, 330))
        
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        
        # Update admin preview if visible
        if self.admin_mode and hasattr(self, 'admin_video_label'):
            try:
                if self.admin_video_label.winfo_exists():
                    self.admin_video_label.imgtk = imgtk
                    self.admin_video_label.configure(image=imgtk)
            except tk.TclError:
                pass
        else:
            # Update main kiosk display
            self.video_label.imgtk = imgtk
            self.video_label.configure(image=imgtk)
    
    def show_toast(self, message, toast_type="info", duration=None):
        """
        Display a temporary notification toast at the bottom of screen.
        
        Args:
            message: Text to display
            toast_type: One of "success", "error", "warning", "info"
            duration: Display time in milliseconds (uses config default if None)
        """
        if duration is None:
            duration = Config.TOAST_DURATION
        
        # Color schemes for each toast type
        colors = {
            "success": (Config.COLOR_GRANTED, "#FFFFFF"),
            "error": (Config.COLOR_DENIED, "#FFFFFF"),
            "warning": (Config.COLOR_WARNING, "#FFFFFF"),
            "info": (Config.COLOR_SCANNING, "#FFFFFF")
        }
        bg_color, fg_color = colors.get(toast_type, colors["info"])
        
        # Create or update toast label
        if not hasattr(self, 'toast_label') or not self.toast_label.winfo_exists():
            self.toast_label = tk.Label(
                self.main_frame,
                text="",
                font=(Config.FONT_FAMILY, 10),
                bg=bg_color,
                fg=fg_color,
                padx=15,
                pady=8
            )
        
        self.toast_label.config(text=message, bg=bg_color, fg=fg_color)
        
        # Position centered near bottom
        self.toast_label.place(relx=0.5, rely=0.92, anchor="center")
        self.toast_label.lift()
        
        # Cancel existing timer
        if self._toast_id:
            self.root.after_cancel(self._toast_id)
        
        # Auto-hide after duration
        self._toast_id = self.root.after(duration, self._hide_toast)
    
    def _hide_toast(self):
        """Remove toast from display."""
        if hasattr(self, 'toast_label') and self.toast_label.winfo_exists():
            self.toast_label.place_forget()
        self._toast_id = None
    
    def grant_access(self, name, confidence):
        """
        Handle successful face recognition - grant access.
        Unlocks door, logs access, and shows success feedback.
        """
        self.set_status("granted", name, confidence)
        self.access_log.add_entry(name, True, confidence)
        self.door_controller.unlock()
        self.update_log_display()
        self._pulse_border(Config.COLOR_GRANTED)
        logger.info(f"Access granted: {name} ({confidence:.1%})")
    
    def deny_access(self):
        """
        Handle failed recognition - deny access.
        Logs attempt and shows denial feedback.
        """
        self.set_status("denied")
        self.access_log.add_entry("Unknown", False, 0.0)
        self.update_log_display()
        self._pulse_border(Config.COLOR_DENIED)
        logger.info("Access denied: Unknown person")
    
    def _pulse_border(self, color, duration=300):
        """Create a brief color pulse on the video container border."""
        original_color = self.video_container.cget('highlightbackground')
        original_thickness = self.video_container.cget('highlightthickness')
        
        self.video_container.config(highlightbackground=color, highlightthickness=3)
        
        self.root.after(
            duration, 
            lambda: self.video_container.config(
                highlightbackground=original_color, 
                highlightthickness=original_thickness
            )
        )
    
    def _cleanup_old_access_entries(self, current_time):
        """
        Remove stale entries from cooldown tracking dict.
        Prevents unbounded memory growth over long uptime.
        """
        max_age = Config.COOLDOWN_SECONDS * 10
        keys_to_remove = [
            name for name, timestamp in self.last_access.items()
            if current_time - timestamp > max_age
        ]
        for key in keys_to_remove:
            del self.last_access[key]
    
    def show_admin_login(self):
        """Display the admin login dialog for authentication."""
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
        
        # Check password using secure hash comparison
        password = result['password']
        if password and verify_password(password, Config.ADMIN_PASSWORD_HASH):
            logger.info("Admin login successful")
            self.show_admin_panel()
        elif password is not None:
            logger.warning("Failed admin login attempt")
            messagebox.showerror("Error", "Invalid password")
    
    def _show_password_keyboard(self, dialog, content, entry):
        """Display on-screen keyboard for password entry."""
        dialog.geometry("480x450")
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() - 480) // 2
        y = (dialog.winfo_screenheight() - 450) // 2
        dialog.geometry(f"480x450+{x}+{y}")
        show_keyboard(content, entry, dialog)
    
    def show_admin_panel(self):
        """
        Display the admin control panel.
        Replaces the main kiosk interface with admin tabs.
        """
        self.admin_mode = True
        self.is_scanning = False
        self.face_system.clear_cache()
        
        # Hide kiosk interface
        self.main_frame.pack_forget()
        
        # Maintain window size
        if Config.FULLSCREEN:
            self.root.attributes('-fullscreen', True)
        else:
            self.root.geometry("480x800")
        
        # Create admin interface
        self.admin_frame = tk.Frame(self.root, bg=Config.COLOR_BG)
        self.admin_frame.pack(fill=tk.BOTH, expand=True)
        
        # Header with title and close button
        header = tk.Frame(self.admin_frame, bg=Config.COLOR_BG)
        header.pack(fill=tk.X, padx=15, pady=(15, 8))
        
        tk.Label(
            header,
            text="Settings",
            font=(Config.FONT_FAMILY, 18, "bold"),
            fg=Config.COLOR_TEXT,
            bg=Config.COLOR_BG
        ).pack(side=tk.LEFT)
        
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
        
        # Configure notebook styling
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
        self.admin_notebook = notebook
        
        # Lock tab switching during operations
        notebook.bind('<<NotebookTabChanged>>', self.on_tab_changed)
        
        # Create admin tabs
        register_tab = tk.Frame(notebook, bg=Config.COLOR_BG)
        notebook.add(register_tab, text="  Register  ")
        self.create_register_tab(register_tab)
        
        train_tab = tk.Frame(notebook, bg=Config.COLOR_BG)
        notebook.add(train_tab, text="  Train  ")
        self.create_train_tab(train_tab)
        
        manage_tab = tk.Frame(notebook, bg=Config.COLOR_BG)
        notebook.add(manage_tab, text="  Users  ")
        self.create_manage_tab(manage_tab)
        
        log_tab = tk.Frame(notebook, bg=Config.COLOR_BG)
        notebook.add(log_tab, text="  Activity  ")
        self.create_log_tab(log_tab)
        
        settings_tab = tk.Frame(notebook, bg=Config.COLOR_BG)
        notebook.add(settings_tab, text="  System  ")
        self.create_settings_tab(settings_tab)
    
    def create_register_tab(self, parent):
        """
        Build the face registration tab with camera preview.
        Supports Face ID style zone-based capture.
        """
        self.reg_main_container = tk.Frame(parent, bg=Config.COLOR_BG)
        self.reg_main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Camera preview with fixed dimensions
        camera_container = tk.Frame(self.reg_main_container, bg=Config.COLOR_CARD, 
                                   highlightbackground=Config.COLOR_BORDER, highlightthickness=1,
                                   width=440, height=330)
        camera_container.pack(pady=(0, 5))
        camera_container.pack_propagate(False)
        
        self.admin_video_label = tk.Label(camera_container, bg="#000000")
        self.admin_video_label.pack(fill=tk.BOTH, expand=True)
        
        # ===== SETUP PANEL (pre-registration) =====
        self.reg_setup_panel = tk.Frame(self.reg_main_container, bg=Config.COLOR_CARD,
                                        highlightbackground=Config.COLOR_BORDER, highlightthickness=1)
        self.reg_setup_panel.pack(fill=tk.X, pady=3)
        
        setup_inner = tk.Frame(self.reg_setup_panel, bg=Config.COLOR_CARD)
        setup_inner.pack(fill=tk.X, padx=10, pady=8)
        
        # Name input row
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
        
        # ===== CAPTURE PANEL (active during registration) =====
        self.reg_capture_panel = tk.Frame(self.reg_main_container, bg=Config.COLOR_CARD,
                                          highlightbackground=Config.COLOR_BORDER, highlightthickness=1)
        
        capture_inner = tk.Frame(self.reg_capture_panel, bg=Config.COLOR_CARD)
        capture_inner.pack(fill=tk.X, padx=10, pady=8)
        
        # Status display row
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
        
        # Bottom row: auto capture button
        btn_row = tk.Frame(capture_inner, bg=Config.COLOR_CARD)
        btn_row.pack(fill=tk.X)
        
        self.auto_capture_btn = tk.Button(
            btn_row,
            text="⟳ Start Auto Capture (100)",
            font=(Config.FONT_FAMILY, 9),
            fg="#FFFFFF",
            bg="#5856D6",
            activebackground="#4744c4",
            activeforeground="#FFFFFF",
            relief=tk.FLAT,
            cursor="hand2",
            command=self.start_auto_capture
        )
        self.auto_capture_btn.pack(side=tk.LEFT, ipady=2, ipadx=8)
    
    def create_train_tab(self, parent):
        """Build the model training tab with progress indicator.
        
        Args:
            parent: Parent frame for the tab content.
        
        Creates a card with dataset statistics, a progress bar for
        training feedback, and a button to trigger full model training.
        """
        # Training card container
        card = tk.Frame(parent, bg=Config.COLOR_CARD, highlightbackground=Config.COLOR_BORDER, highlightthickness=1)
        card.pack(fill=tk.X, padx=10, pady=10)
        
        inner = tk.Frame(card, bg=Config.COLOR_CARD)
        inner.pack(fill=tk.X, padx=15, pady=15)
        
        # Section header
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
        
        # Dataset statistics display
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
        
        # Progress bar with custom styling
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
        
        # Training status label
        self.train_status_label = tk.Label(
            inner,
            text="Ready to train",
            font=(Config.FONT_FAMILY, 9),
            fg=Config.COLOR_TEXT_SECONDARY,
            bg=Config.COLOR_CARD
        )
        self.train_status_label.pack(pady=(0, 10))
        
        # Train button - triggers full dataset encoding
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
        """Build the user management tab with list and delete functionality.
        
        Args:
            parent: Parent frame for the tab content.
        
        Displays a list of registered users with their photo counts
        and provides a delete button for removing users.
        """
        # Header with title and refresh button
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
        
        # User list container
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
        
        self.refresh_manage_list()
        
        # Action buttons
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
        """Build the access log tab with filtering capabilities."""
        # Header with title and clear button
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
        
        # Quick date filter buttons
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
        
        # Log entries list
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
        
        # Filter state
        self.log_date_from = None
        self.log_date_to = None
        self.log_name_var = tk.StringVar(value="All")
        
        # Load initial entries
        self.populate_log_listbox(self.access_log.get_recent(100))
    
    def set_log_date_range(self, days_back):
        """Apply a quick date filter to the access log display."""
        from datetime import timedelta, date
        today = datetime.now().date()
        
        if days_back == 0:
            date_from = today
        else:
            date_from = today - timedelta(days=days_back)
        
        entries = self.access_log.get_filtered(date_from, today, None, count=100)
        self.populate_log_listbox(entries)
    
    def clear_log_filter(self):
        """Remove all filters and show all log entries."""
        self.log_name_var.set("All")
        self.populate_log_listbox(self.access_log.get_recent(100))
    
    def populate_log_listbox(self, entries):
        """Fill the log listbox with formatted access entries."""
        self.admin_log_listbox.delete(0, tk.END)
        
        if not entries:
            self.admin_log_listbox.insert(tk.END, "  No entries found")
            return
        
        for entry in entries:
            timestamp = datetime.fromisoformat(entry['timestamp']).strftime("%b %d, %H:%M:%S")
            status_icon = "✓" if entry['access_granted'] else "✕"
            confidence = entry.get('confidence', 0)
            conf_str = f"{confidence:.0%}" if confidence > 0 else "—"
            name = entry['name'][:15] if len(entry['name']) > 15 else entry['name']
            self.admin_log_listbox.insert(
                tk.END, 
                f"  {status_icon}  {timestamp}   {name:<15}  {conf_str:>5}"
            )
    
    def create_settings_tab(self, parent):
        """Build the system settings tab with configuration options."""
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
        
        # Display current recognition settings as read-only info
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
        
        # System Information Card
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
        
        # Detect hardware configuration
        camera_type = "Pi Camera" if USE_PICAMERA else "USB Webcam"
        gpio_status = "Hardware" if USE_GPIO else "Simulated"
        
        system_items = [
            ("Camera", camera_type),
            ("Door Control", gpio_status),
            ("Backup", "Enabled" if Config.BACKUP_ENABLED else "Disabled"),
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
        
        # Exit kiosk mode button (requires admin authentication to reach this tab)
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
    
    # ==================== USER MANAGEMENT METHODS ====================
    
    def refresh_manage_list(self):
        """Refresh the registered users list in the manage tab.
        
        Shows users from the trained model (pickle file), not the dataset folder.
        Maintains an index-to-name mapping for deletion operations.
        """
        self.manage_listbox.delete(0, tk.END)
        self.person_map = {}
        
        # Get trained persons from pickle file (not dataset folder)
        trained_names = self.face_system.get_trained_persons()
        
        # Count encodings per person for display
        name_counts = {}
        for name in self.face_system.known_names:
            name_counts[name] = name_counts.get(name, 0) + 1
        
        for idx, name in enumerate(sorted(trained_names)):
            self.person_map[idx] = name
            count = name_counts.get(name, 0)
            self.manage_listbox.insert(tk.END, f"  {name}   •   {count} encodings")
    
    def refresh_train_tab(self):
        """Update the training tab's dataset statistics display."""
        if hasattr(self, 'dataset_info_label'):
            persons = self.face_system.get_registered_persons()
            total_images = sum(count for _, count in persons)
            self.dataset_info_label.config(text=f"{len(persons)} people  •  {total_images} photos")
    
    # ==================== REGISTRATION WORKFLOW ====================
    
    def start_registration(self):
        """Begin the face registration workflow for a new person.
        
        Validates the entered name, checks for duplicates (with user
        confirmation for adding more photos), and transitions the UI
        from setup mode to capture mode.
        """
        name = self.reg_name_entry.get().strip()
        if not name:
            self.show_toast("Please enter a name", "warning")
            self.reg_name_entry.focus_set()
            return
        
        # Check for duplicate names with case-insensitive comparison
        existing_persons = [p[0].lower() for p in self.face_system.get_registered_persons()]
        if name.lower() in existing_persons:
            if not messagebox.askyesno(
                "Name Exists", 
                f"'{name}' already exists. Add more photos to this person?"
            ):
                return
        
        # Initialize registration state
        self.registration_mode = True
        self.registration_name = name
        self.captured_count = 0
        self.auto_capture_mode = False
        self.zone_captures = {'center': 0, 'left': 0, 'right': 0, 'up': 0, 'down': 0}
        self.reg_process_locked = True  # Lock tab navigation during registration
        
        # Transition UI from setup panel to capture panel
        self.reg_setup_panel.pack_forget()
        self.reg_capture_panel.pack(fill=tk.X, pady=5)
        
        # Update capture panel labels
        self.reg_name_display.config(text=name)
        self.reg_count_label.config(text="0 photos • Position face in frame")
        
        self.show_toast(f"Registering {name}", "info")
    
    def stop_registration(self):
        """End the registration session and optionally trigger auto-training.
        
        Prevents stopping during active auto-capture/training operations.
        Resets all registration state and transitions UI back to setup mode.
        Triggers single-person training if auto-train option is enabled.
        """
        # Only block if auto-capture is actively running (not manual capture)
        if self.auto_capture_mode and self.reg_process_locked:
            messagebox.showwarning("Please Wait", "Auto-capture or encoding in progress. Please wait for completion.")
            return
        
        # Save registration info before resetting
        person_name = self.registration_name
        captured = self.captured_count
        
        # Reset all registration state
        self.registration_mode = False
        self.registration_name = ""
        self.auto_capture_mode = False
        self.zone_captures = {'center': 0, 'left': 0, 'right': 0, 'up': 0, 'down': 0}
        self.reg_process_locked = False  # Unlock tab navigation
        
        # Transition UI back to setup panel
        self.reg_capture_panel.pack_forget()
        self.reg_setup_panel.pack(fill=tk.X, pady=5)
        
        # Reset auto capture button for next registration
        self.auto_capture_btn.config(text="⟳ Start Auto Capture (100)", bg="#5856D6", command=self.start_auto_capture)
        
        # Clear name entry for next use
        self.reg_name_entry.delete(0, tk.END)
        
        # Refresh lists to show new person
        self.refresh_manage_list()
        self.refresh_train_tab()
        
        # Show completion feedback
        if captured > 0:
            self.show_toast(f"Captured {captured} photos for {person_name}", "success")
        
        # Auto-train if not already trained via auto-capture flow
        already_trained = getattr(self, 'already_trained', False)
        self.already_trained = False
        if not already_trained and captured > 0 and person_name:
            self.train_single_person(person_name)
    
    # ==================== AUTO-CAPTURE (FACE ID STYLE) ====================
    
    def start_auto_capture(self):
        """Begin automatic Face ID style photo capture.
        
        Enables zone-based capture mode that guides the user to move
        their face to different positions (center, left, right, up, down)
        to capture diverse angles for better recognition accuracy.
        """
        if not self.registration_mode:
            return
        
        # Enable auto-capture and lock navigation
        self.auto_capture_mode = True
        self.reg_process_locked = True
        self.last_auto_capture = time.time()
        self.zone_captures = {'center': 0, 'left': 0, 'right': 0, 'up': 0, 'down': 0}
        
        # Update button state
        self.auto_capture_btn.config(text="⏹ Stop", bg=Config.COLOR_DENIED, command=self.stop_auto_capture)
        self.reg_count_label.config(text="Move face to fill all zones...")
    
    def stop_auto_capture(self):
        """Stop the automatic capture process with user confirmation.
        
        If process is locked (capture in progress), requires confirmation.
        Photos already captured are kept but training won't auto-start.
        """
        if self.reg_process_locked:
            if not messagebox.askyesno("Cancel Capture?", 
                "Are you sure you want to cancel?\nPhotos already captured will be kept but training won't start."):
                return
        
        # Reset auto-capture state
        self.auto_capture_mode = False
        self.reg_process_locked = False
        self.auto_capture_btn.config(text="⟳ Start Auto Capture (100)", bg="#5856D6", command=self.start_auto_capture)
        self.update_registration_ui()
    
    def draw_faceid_overlay(self, frame):
        """Draw Face ID style visual guide showing zone positions.
        
        Args:
            frame: OpenCV frame to draw the overlay on.
        
        Draws a circular face positioning guide with zone indicators
        around it (up, down, left, right, center) to help users
        position their face correctly during auto-capture.
        """
        frame_height, frame_width = frame.shape[:2]
        center_x, center_y = frame_width // 2, frame_height // 2
        
        # Calculate guide circle radius based on frame size
        guide_radius = min(frame_width, frame_height) // 4
        
        # Draw outer positioning guide circle
        cv2.circle(frame, (center_x, center_y), guide_radius, (100, 100, 100), 2)
        
        # Zone indicator positions around the guide circle
        zone_positions = {
            'up': (center_x, center_y - guide_radius - 30),
            'down': (center_x, center_y + guide_radius + 30),
            'left': (center_x - guide_radius - 30, center_y),
            'right': (center_x + guide_radius + 30, center_y),
            'center': (center_x, center_y)
        }
        
        # Draw zone indicators showing capture progress
        for zone, (zx, zy) in zone_positions.items():
            current = self.zone_captures.get(zone, 0)
            target = self.zone_targets.get(zone, 0)
            
            # Calculate completion percentage for this zone
            fill_pct = min(1.0, current / target) if target > 0 else 1.0
            
            # Color coding: gray=empty, yellow=partial, green=complete
            if fill_pct >= 1.0:
                color = (0, 255, 0)   # Green - zone complete
            elif fill_pct > 0:
                color = (0, 255, 255) # Yellow - in progress
            else:
                color = (80, 80, 80)  # Gray - not started
            
            # Highlight the current target zone with larger radius
            radius = 18 if zone == self.current_zone else 12
            thickness = -1 if fill_pct >= 1.0 else 2  # Filled if complete
            
            if zone == 'center':
                # Center uses a ring instead of filled circle
                cv2.circle(frame, (zx, zy), radius + 5, color, 2)
            else:
                cv2.circle(frame, (zx, zy), radius, color, thickness)
                # Show count for incomplete zones
                if fill_pct < 1.0:
                    cv2.putText(frame, str(current), (zx - 8, zy + 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Mode header text
        mode_text = "FACE ID CAPTURE" if self.auto_capture_mode else "REGISTRATION"
        cv2.putText(frame, mode_text, (15, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 100, 200), 2)
        cv2.putText(frame, self.registration_name, (15, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Progress bar at bottom (only during auto-capture)
        if self.auto_capture_mode:
            bar_width = 350
            bar_height = 14
            bar_x = center_x - bar_width // 2
            bar_y = frame_height - 45
            
            progress = self.captured_count / self.auto_capture_target
            
            # Draw progress bar background
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (40, 40, 40), -1)
            # Draw progress fill
            fill_width = int(bar_width * progress)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height), (0, 200, 0), -1)
            # Draw border
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height), (100, 100, 100), 1)
            
            # Progress count text
            count_text = f"{self.captured_count}/{self.auto_capture_target}"
            cv2.putText(frame, count_text, (bar_x + bar_width + 10, bar_y + 12),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Zone guidance text
            zones_complete = sum(1 for z in self.zone_captures if self.zone_captures[z] >= self.zone_targets.get(z, 0))
            if zones_complete < 5:
                # Find the next zone that needs more captures
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
        """Update the registration UI labels with current capture progress."""
        if self.auto_capture_mode:
            zones_done = sum(1 for z in self.zone_captures 
                           if self.zone_captures[z] >= self.zone_targets.get(z, 0))
            self.reg_count_label.config(
                text=f"{self.captured_count} photos • {zones_done}/5 zones"
            )
        else:
            self.reg_count_label.config(text=f"{self.captured_count} photos")
    
    def complete_auto_registration(self):
        """Complete auto-capture and automatically begin training.
        
        Called when the capture target is reached. Transitions the UI
        to training mode and initiates single-person encoding. The
        process lock remains active until training completes.
        """
        self.auto_capture_mode = False
        # Keep reg_process_locked True - will be cleared after encoding completes
        
        zones_done = sum(1 for z in self.zone_captures 
                        if self.zone_captures[z] >= self.zone_targets.get(z, 0))
        
        # Update UI to show training state
        self.reg_count_label.config(
            text=f"✓ {self.captured_count} photos • Training..."
        )
        
        # Disable button during training
        self.auto_capture_btn.config(text="Encoding...", bg="#888888", state=tk.DISABLED)
        self.stop_reg_btn.config(state=tk.DISABLED)
        
        # Start training in background thread
        self.train_single_person_with_progress(self.registration_name)
    
    # ==================== MODEL TRAINING ====================
    
    def train_single_person_with_progress(self, person_name):
        """Train encodings for a single person with UI progress updates.
        
        Args:
            person_name: Name of the person to encode.
        
        Runs encoding in a background thread while updating the UI
        with progress. The reg_process_locked flag prevents navigation
        until encoding and saving completes.
        """
        self.is_training = True
        self.reg_process_locked = True
        total_images = 0
        
        def update_progress_label(text):
            """Thread-safe label update helper."""
            try:
                if hasattr(self, 'reg_count_label') and self.reg_count_label.winfo_exists():
                    self.reg_count_label.config(text=text)
            except tk.TclError:
                pass
        
        def training_thread():
            nonlocal total_images
            # Count images for progress tracking
            from imutils import paths
            person_folder = os.path.join(self.face_system.dataset_path, person_name)
            if os.path.exists(person_folder):
                total_images = len(list(paths.list_images(person_folder)))
            
            def progress_callback(current, total, filepath):
                """Update UI with encoding progress."""
                if current == total:
                    self.root.after(0, lambda: update_progress_label("Saving encodings..."))
                else:
                    self.root.after(0, lambda c=current, t=total: update_progress_label(f"Encoding {c}/{t}..."))
            
            success, message = self.face_system.train_single_person(person_name, progress_callback)
            # Schedule completion on main thread
            self.root.after(0, lambda: self.training_with_progress_complete(success, message))
        
        thread = threading.Thread(target=training_thread, daemon=True)
        thread.start()
    
    def training_with_progress_complete(self, success, message):
        """Handle completion of training from the auto-capture flow.
        
        Args:
            success: Whether training succeeded.
            message: Status message from the training operation.
        
        Releases the process lock and re-enables UI controls.
        """
        self.is_training = False
        self.reg_process_locked = False  # Release lock - encoding complete
        
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
                # Prevent double-training when stop_registration is called
                self.already_trained = True
                # Auto-close registration after short delay
                self.root.after(1500, self.stop_registration)
        except tk.TclError:
            pass  # Widgets were destroyed
    
    def train_single_person(self, person_name):
        """Train encodings for a single person without progress UI.
        
        Args:
            person_name: Name of the person to encode.
        
        Used for manual training triggers (not from auto-capture flow).
        Runs encoding in a background thread to keep UI responsive.
        """
        # Update label if widget exists
        try:
            if hasattr(self, 'reg_count_label') and self.reg_count_label.winfo_exists():
                self.reg_count_label.config(text=f"Training {person_name}...")
        except tk.TclError:
            pass
        
        self.is_training = True  # Pause live recognition during training
        
        def training_thread():
            success, message = self.face_system.train_single_person(person_name)
            self.root.after(0, lambda: self.single_training_complete(success, message))
        
        thread = threading.Thread(target=training_thread, daemon=True)
        thread.start()
    
    def single_training_complete(self, success, message):
        """Handle completion of single-person training.
        
        Args:
            success: Whether training succeeded.
            message: Status message from the training operation.
        """
        self.is_training = False  # Resume live recognition
        
        # Update label if widget exists
        try:
            if hasattr(self, 'reg_count_label') and self.reg_count_label.winfo_exists():
                if success:
                    self.reg_count_label.config(text=f"✓ {message}")
                else:
                    self.reg_count_label.config(text=f"✗ Training failed")
                # Reset label after delay
                self.root.after(3000, self._reset_reg_count_label)
        except tk.TclError:
            pass
        
        if success:
            messagebox.showinfo("Training Complete", message)
        else:
            messagebox.showerror("Training Failed", message)
    
    def _reset_reg_count_label(self):
        """Reset the registration count label to default state."""
        try:
            if hasattr(self, 'reg_count_label') and self.reg_count_label.winfo_exists():
                self.reg_count_label.config(text="0 photos")
        except tk.TclError:
            pass
    
    def start_training(self):
        """Begin full model training for all registered persons.
        
        Processes all images in the dataset folder, generates face
        embeddings, and saves them to the encodings file. Updates
        the progress bar and status label during training.
        """
        self.train_btn.config(state=tk.DISABLED)
        self.train_progress['value'] = 0
        self.train_status_label.config(text="Training in progress...")
        self.is_training = True  # Pause live recognition during training
        self.reg_process_locked = True  # Lock tab navigation during training
        
        def training_thread():
            def progress_callback(current, total, filepath):
                """Update progress bar and status from training thread."""
                progress = (current / total) * 100
                self.root.after(0, lambda: self.train_progress.configure(value=progress))
                self.root.after(0, lambda: self.train_status_label.config(
                    text=f"Processing {current}/{total}..."))
            
            success, message = self.face_system.train_model(progress_callback)
            self.root.after(0, lambda: self.training_complete(success, message))
        
        thread = threading.Thread(target=training_thread, daemon=True)
        thread.start()
    
    def training_complete(self, success, message):
        """Handle completion of full model training.
        
        Args:
            success: Whether training succeeded.
            message: Status message from the training operation.
        """
        self.is_training = False  # Resume live recognition
        self.reg_process_locked = False  # Unlock tab navigation
        self.train_btn.config(state=tk.NORMAL)
        self.train_progress['value'] = 100 if success else 0
        self.train_status_label.config(text=message if len(message) < 50 else message[:47] + "...")
        
        if success:
            self.show_toast("Training completed successfully!", "success")
            self.update_info_label()
        else:
            self.show_toast(f"Training failed: {message}", "error")
    
    # ==================== USER DELETION ====================
    
    def delete_person(self):
        """Delete the selected person from the trained model.
        
        Removes the person's encodings from the saved model but keeps
        the original photos in the dataset folder. Uses person_map for 
        robust name lookup to avoid string parsing issues with display text.
        """
        selection = self.manage_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a person to delete")
            return
        
        selected_index = selection[0]
        
        # Use person_map for robust name lookup
        name = self.person_map.get(selected_index)
        if name is None:
            messagebox.showerror("Error", "Could not identify selected person. Please refresh and try again.")
            return
        
        if messagebox.askyesno("Confirm Delete", f"Remove '{name}' from recognition model?\n\n(Photos will be kept in dataset folder)"):
            # Remove encodings from trained model only (keep photos)
            success, message = self.face_system.remove_person_from_model(name)
            if success:
                logger.info(f"User removed from model: {message}")
            else:
                logger.warning(f"Delete warning: {message}")
            
            # Refresh UI lists
            self.refresh_manage_list()
            self.update_info_label()
            
            self.show_toast(f"'{name}' removed from recognition", "success")
    
    # ==================== ACCESS LOG MANAGEMENT ====================
    
    def clear_access_log(self):
        """Clear all access log entries with user confirmation."""
        entry_count = len(self.access_log.entries)
        if entry_count == 0:
            self.show_toast("Log is already empty", "info")
            return
        
        if messagebox.askyesno(
            "Clear Access Log", 
            f"Delete all {entry_count} log entries?\n\nThis action cannot be undone."
        ):
            self.access_log.clear()
            self.admin_log_listbox.delete(0, tk.END)
            self.admin_log_listbox.insert(tk.END, "  No entries found")
            self.update_log_display()
            self.show_toast("Access log cleared", "success")
    
    def update_info_label(self):
        """Update the user count display in the kiosk interface."""
        count = len(self.face_system.get_trained_persons())
        self.info_label.config(text=f"{count} Users")
    
    # ==================== ADMIN PANEL NAVIGATION ====================
    
    def on_tab_changed(self, event):
        """Handle notebook tab changes with process lock enforcement.
        
        Prevents tab switching during active capture or training operations
        to ensure data integrity and process completion.
        """
        if self.reg_process_locked:
            # Determine which tab to force back to based on what's in progress
            if self.registration_mode or self.auto_capture_mode:
                # Registration/auto-capture in progress - force back to register tab (index 0)
                self.admin_notebook.select(0)
                messagebox.showwarning("Registration in Progress", "Please wait for capture and encoding to complete.")
            elif self.is_training:
                # Full model training from Train tab - force back to train tab (index 1)
                self.admin_notebook.select(1)
                messagebox.showwarning("Training in Progress", "Please wait for training to complete.")
            else:
                # Fallback - force to register tab
                self.admin_notebook.select(0)
                messagebox.showwarning("Process in Progress", "Please wait for the current operation to complete.")
    
    def close_admin_panel(self):
        """Close the admin panel and restore the kiosk interface.
        
        Prevents closing during active capture or training operations.
        Cleans up registration state if a session was in progress.
        """
        # Prevent closing during active process
        if self.reg_process_locked:
            messagebox.showwarning("Process in Progress", "Please wait for capture and encoding to complete.")
            return
        
        # Clean up any in-progress registration
        if self.registration_mode:
            self.stop_registration()
        
        # Reset mode flags
        self.admin_mode = False
        self.is_scanning = True
        
        # Clear recognition cache and stats for fresh scanning
        self.face_system.clear_cache()
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Destroy admin frame and restore main kiosk UI
        self.admin_frame.destroy()
        
        # Ensure window size remains consistent
        if Config.FULLSCREEN:
            self.root.attributes('-fullscreen', True)
        else:
            self.root.geometry("480x800")
        
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Refresh displays with any updates from admin operations
        self.update_log_display()
        self.update_info_label()
    
    def exit_kiosk(self):
        """Exit the kiosk application after user confirmation."""
        if messagebox.askyesno("Exit", "Are you sure you want to exit the kiosk?"):
            self.on_closing()
    
    # ==================== APPLICATION CLEANUP ====================
    
    def on_closing(self):
        """Clean up resources and close the application gracefully.
        
        Stops background threads, cleans up GPIO, and destroys the window.
        Called when the window is closed or exit is requested.
        """
        # Signal all loops to stop
        self.is_running = False
        self.is_scanning = False
        
        # Stop the background recognition thread
        self.face_system.stop_recognition_thread()
        
        # Wait for camera thread to finish
        if self.camera_thread:
            self.camera_thread.join(timeout=2)
        
        # Release GPIO resources
        self.door_controller.cleanup()
        
        # Destroy the tkinter window
        self.root.destroy()


# ==================== APPLICATION ENTRY POINT ====================

def main():
    """Application entry point.
    
    Validates configuration, initializes the Tkinter root window,
    creates the kiosk application, and starts the main event loop.
    Configuration errors are logged and displayed before exit.
    """
    # Validate configuration before starting
    config_errors = Config.validate()
    if config_errors:
        logger.error("Configuration validation failed:")
        for error in config_errors:
            logger.error(f"  - {error}")
        print("\nConfiguration errors detected. Please fix the following issues:")
        for error in config_errors:
            print(f"  - {error}")
        return
    
    # Log startup information
    logger.info("Door Entry Kiosk starting...")
    logger.info(f"Camera: {'Pi Camera' if USE_PICAMERA else 'USB Webcam'}")
    logger.info(f"Door Control: {'GPIO Hardware' if USE_GPIO else 'Simulated'}")
    logger.info(f"Backup: {'Enabled' if Config.BACKUP_ENABLED else 'Disabled'}")
    
    # Create and run the application
    root = tk.Tk()
    app = DoorEntryKiosk(root)
    root.mainloop()


if __name__ == "__main__":
    main()
