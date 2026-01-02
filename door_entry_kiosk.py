# ==================== STANDARD LIBRARY IMPORTS ====================
import tkinter as tk
from tkinter import ttk, messagebox
import cv2
import os
import threading
from queue import Queue, Empty
from multiprocessing import Process, Queue as MPQueue, Event as MPEvent
import time
from datetime import datetime
import pickle
import numpy as np
from PIL import Image, ImageTk
import json
import sqlite3
import gc
import logging
import shutil
from typing import Optional, List, Tuple, Dict, Any, Callable
from abc import ABC, abstractmethod

# ==================== SECURITY IMPORTS ====================
# bcrypt: Secure password hashing (slow by design to resist brute-force)
try:
    import bcrypt
    BCRYPT_AVAILABLE = True
except ImportError:
    BCRYPT_AVAILABLE = False
    logging.warning("bcrypt not installed. Install with: pip install bcrypt")

# python-dotenv: Load secrets from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()  # Load .env file if present
except ImportError:
    pass  # dotenv is optional, can use system environment variables

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
    # Suppress FutureWarnings from insightface (rcond and estimate deprecation)
    import warnings
    warnings.filterwarnings('ignore', category=FutureWarning, module='insightface')
    warnings.filterwarnings('ignore', category=FutureWarning, module='skimage')
    
    from insightface.app import FaceAnalysis
    import insightface
    
    # Get available ONNX Runtime providers (avoid CUDA warning on CPU-only systems)
    import onnxruntime as ort
    AVAILABLE_PROVIDERS = ort.get_available_providers()
    # Prefer CUDA if available, otherwise use CPU
    ONNX_PROVIDERS = [p for p in ['CUDAExecutionProvider', 'CPUExecutionProvider'] if p in AVAILABLE_PROVIDERS]
    if not ONNX_PROVIDERS:
        ONNX_PROVIDERS = ['CPUExecutionProvider']
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

# FAISS: Optional - enables fast vector similarity search for large user databases
USE_FAISS = False
try:
    import faiss
    USE_FAISS = True
    logger.info("FAISS available - Fast vector search enabled")
except ImportError:
    USE_FAISS = False
    logger.info("FAISS not available - using linear search (install with: pip install faiss-cpu)")

# MediaPipe: Optional - enables blink-based liveness detection
USE_MEDIAPIPE = False
try:
    import mediapipe as mp
    USE_MEDIAPIPE = True
    logger.info("MediaPipe available - Blink liveness detection enabled")
except ImportError:
    USE_MEDIAPIPE = False
    logger.info("MediaPipe not available - liveness detection disabled (install with: pip install mediapipe)")


# ==================== SECURITY UTILITIES ====================
def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt for secure storage.
    
    bcrypt is designed to be slow, making brute-force attacks infeasible.
    It automatically generates and embeds a unique salt in the hash.
    
    Args:
        password: The plaintext password to hash
        
    Returns:
        The bcrypt hash string (includes salt and cost factor)
        
    Raises:
        RuntimeError: If bcrypt is not installed
    """
    if not BCRYPT_AVAILABLE:
        raise RuntimeError("bcrypt is required for password hashing. Install with: pip install bcrypt")
    
    # Generate salt and hash password (cost factor 12 = ~250ms on modern hardware)
    salt = bcrypt.gensalt(rounds=12)
    return bcrypt.hashpw(password.encode('utf-8'), salt).decode('utf-8')

def verify_password(password: str, hashed: str) -> bool:
    """
    Verify a plaintext password against a stored bcrypt hash.
    
    Uses constant-time comparison to prevent timing attacks.
    
    Args:
        password: The plaintext password to verify
        hashed: The stored bcrypt hash to compare against
        
    Returns:
        True if the password matches, False otherwise
    """
    if not BCRYPT_AVAILABLE:
        logging.error("bcrypt is required for password verification")
        return False
    
    try:
        return bcrypt.checkpw(password.encode('utf-8'), hashed.encode('utf-8'))
    except (ValueError, TypeError) as e:
        logging.error(f"Password verification failed: {e}")
        return False


# ==================== APPLICATION CONFIGURATION ====================
class Config:
    """Central configuration class for all application settings.
    
    Modify these values to customize the kiosk behavior.
    """
    
    # ----- Window Settings -----
    FULLSCREEN = False                    # Set True for production kiosk deployment
    WINDOW_TITLE = "Door Entry System"    # Window title bar text
    
    # ----- Security Settings -----
    # Load admin password hash from environment variable
    # Generate a new hash with: python -c "import bcrypt; print(bcrypt.hashpw(b'your_password', bcrypt.gensalt(12)).decode())"
    ADMIN_PASSWORD_HASH = os.environ.get('ADMIN_PASSWORD_HASH', '')
    
    # Warn if no password hash is configured
    if not ADMIN_PASSWORD_HASH:
        logging.warning(
            "ADMIN_PASSWORD_HASH not set in environment. "
            "Create a .env file with ADMIN_PASSWORD_HASH=<bcrypt_hash> or set the environment variable. "
            "Generate hash with: python -c \"import bcrypt; print(bcrypt.hashpw(b'your_password', bcrypt.gensalt(12)).decode())\""
        )
    
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
    
    # ----- Power/Compute Optimization -----
    IDLE_FPS = 10                          # Lower frame rate when no faces detected
    ACTIVE_FPS = 30                        # Higher frame rate when faces detected
    IDLE_RECOGNITION_INTERVAL = 4          # Process every Nth frame when idle (less frequent)
    CAPTURE_THREAD_SLEEP_MS = 5            # Milliseconds to sleep between captures (reduce CPU spin)
    GC_INTERVAL_SECONDS = 120              # Garbage collection interval (seconds)
    USE_MULTIPROCESSING = True             # Use separate process for recognition (bypasses GIL)
    FAISS_INDEX_THRESHOLD = 50             # Use FAISS index when user count exceeds this
    
    # ----- Liveness Detection (Blink) -----
    ENABLE_BLINK_DETECTION = True          # Enable blink-based liveness detection
    EAR_THRESHOLD = 0.30                   # Eye Aspect Ratio threshold for blink detection (higher = more sensitive)
    BLINK_CONSEC_FRAMES = 1                # Consecutive frames below threshold to count as blink (1 = most responsive)
    BLINK_REQUIRED_COUNT = 1               # Number of blinks required to pass liveness
    BLINK_TIMEOUT_SECONDS = 5.0            # Time window to detect required blinks
    
    # ----- Memory Management Settings -----
    MAX_CACHE_ENTRIES = 20  # Maximum faces to keep in cache
    ACCESS_LOG_MAX_MEMORY_ENTRIES = 1000  # Max entries to keep in memory (older are on disk only)
    FRAME_POOL_SIZE = 3  # Number of pre-allocated frames for recognition queue
    LOG_PAGE_SIZE = 50  # Number of log entries per page in admin panel
    USERS_PAGE_SIZE = 20  # Number of users per page in admin panel
    
    # Door Control (GPIO Pin for Raspberry Pi)
    DOOR_RELAY_PIN = 17
    DOOR_UNLOCK_DURATION = 1  # Seconds to keep door unlocked
    
    # File Paths
    DATASET_PATH = "dataset"
    ENCODINGS_PATH = "encodings.pickle"
    DISABLED_ENCODINGS_PATH = "disabled_encodings.pickle"  # Revoked users stored here
    ACCESS_LOG_PATH = "access_log.db"  # SQLite database for atomic writes
    
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
        except (OSError, IOError, shutil.Error) as e:
            logger.error(f"Failed to create backup of {file_path}: {e}", exc_info=True)
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
            except OSError as e:
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
        except (OSError, IOError, shutil.Error) as e:
            logger.error(f"Failed to restore from backup: {e}", exc_info=True)
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


# ==================== HARDWARE ABSTRACTION LAYER ====================
class HardwareInterface(ABC):
    """
    Abstract base class for hardware control.
    
    Provides a clean interface for door control operations,
    allowing different implementations for real hardware vs simulation.
    """
    
    @abstractmethod
    def initialize(self) -> None:
        """Initialize hardware resources."""
        pass
    
    @abstractmethod
    def unlock_door(self, duration: float) -> None:
        """Unlock the door for specified duration in seconds."""
        pass
    
    @abstractmethod
    def is_door_unlocked(self) -> bool:
        """Check if door is currently unlocked."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Release hardware resources."""
        pass


class RealHardware(HardwareInterface):
    """
    Real GPIO-based hardware control for Raspberry Pi.
    Controls physical door relay via GPIO pins.
    """
    
    def __init__(self, relay_pin: int = Config.DOOR_RELAY_PIN):
        self.relay_pin = relay_pin
        self._is_unlocked = False
        self._unlock_thread: Optional[threading.Thread] = None
    
    def initialize(self) -> None:
        """Set up GPIO pins for relay control."""
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.relay_pin, GPIO.OUT)
        GPIO.output(self.relay_pin, GPIO.LOW)  # Start locked
        logger.info(f"GPIO initialized on pin {self.relay_pin}")
    
    def unlock_door(self, duration: float) -> None:
        """Activate relay to unlock door for specified duration."""
        if self._unlock_thread and self._unlock_thread.is_alive():
            return
        
        self._unlock_thread = threading.Thread(
            target=self._unlock_sequence, 
            args=(duration,),
            daemon=True
        )
        self._unlock_thread.start()
    
    def _unlock_sequence(self, duration: float) -> None:
        """Execute timed unlock sequence in separate thread."""
        self._is_unlocked = True
        GPIO.output(self.relay_pin, GPIO.HIGH)
        logger.info(f"Door unlocked for {duration} seconds")
        
        time.sleep(duration)
        
        GPIO.output(self.relay_pin, GPIO.LOW)
        self._is_unlocked = False
        logger.info("Door locked")
    
    def is_door_unlocked(self) -> bool:
        """Return current door lock state."""
        return self._is_unlocked
    
    def cleanup(self) -> None:
        """Release GPIO resources."""
        GPIO.cleanup()
        logger.info("GPIO cleanup complete")


class MockHardware(HardwareInterface):
    """
    Simulated hardware for development and testing.
    Logs actions without requiring actual GPIO hardware.
    """
    
    def __init__(self):
        self._is_unlocked = False
        self._unlock_thread: Optional[threading.Thread] = None
    
    def initialize(self) -> None:
        """Simulated initialization - just log."""
        logger.info("Mock hardware initialized (simulation mode)")
    
    def unlock_door(self, duration: float) -> None:
        """Simulate door unlock with timing."""
        if self._unlock_thread and self._unlock_thread.is_alive():
            return
        
        self._unlock_thread = threading.Thread(
            target=self._unlock_sequence,
            args=(duration,),
            daemon=True
        )
        self._unlock_thread.start()
    
    def _unlock_sequence(self, duration: float) -> None:
        """Simulate timed unlock sequence."""
        self._is_unlocked = True
        logger.info(f"[SIMULATED] Door unlocked for {duration} seconds")
        
        time.sleep(duration)
        
        self._is_unlocked = False
        logger.info("[SIMULATED] Door locked")
    
    def is_door_unlocked(self) -> bool:
        """Return simulated door lock state."""
        return self._is_unlocked
    
    def cleanup(self) -> None:
        """Simulated cleanup - nothing to release."""
        logger.info("Mock hardware cleanup complete")


class DoorController:
    """
    High-level door control interface.
    
    Automatically selects RealHardware or MockHardware based on
    GPIO availability. Provides a consistent API regardless of
    the underlying hardware implementation.
    """
    
    def __init__(self):
        """Initialize door controller with appropriate hardware backend."""
        if USE_GPIO:
            self.hardware: HardwareInterface = RealHardware()
        else:
            self.hardware: HardwareInterface = MockHardware()
        
        self.hardware.initialize()
    
    def unlock(self, duration: Optional[float] = None) -> None:
        """
        Unlock the door for a specified duration.
        
        Args:
            duration: Seconds to keep door unlocked (uses config default if None)
        """
        if duration is None:
            duration = Config.DOOR_UNLOCK_DURATION
        self.hardware.unlock_door(duration)
    
    @property
    def is_unlocked(self) -> bool:
        """Check if door is currently unlocked."""
        return self.hardware.is_door_unlocked()
    
    def cleanup(self) -> None:
        """Release hardware resources on application exit."""
        self.hardware.cleanup()


# ==================== ACCESS LOG ====================
class AccessLog:
    """
    Manages access event logging with persistent SQLite storage.
    
    Stores access attempts (granted/denied) with timestamps using SQLite
    for atomic writes (safe against power loss) and efficient SQL queries.
    Maintains a small in-memory cache for fast recent entry access.
    """
    
    def __init__(self, log_path: Optional[str] = None):
        """
        Initialize access log with SQLite database.
        
        Args:
            log_path: Path to SQLite database file (uses config default if None)
        """
        self.log_path = log_path or Config.ACCESS_LOG_PATH
        self.max_memory_entries = getattr(Config, 'ACCESS_LOG_MAX_MEMORY_ENTRIES', 1000)
        self._cache: List[Dict[str, Any]] = []  # Recent entries cache
        self._cache_dirty = False
        self._init_database()
        self._load_cache()
    
    def _get_connection(self) -> sqlite3.Connection:
        """
        Get a database connection with optimized settings.
        
        Returns:
            SQLite connection with WAL mode for concurrent reads
        """
        conn = sqlite3.connect(self.log_path, timeout=10.0)
        conn.row_factory = sqlite3.Row  # Enable dict-like row access
        # WAL mode for better concurrent access and crash recovery
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")  # Balance between safety and speed
        return conn
    
    def _init_database(self) -> None:
        """Create the access_log table if it doesn't exist."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS access_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    name TEXT NOT NULL,
                    access_granted INTEGER NOT NULL,
                    confidence REAL NOT NULL
                )
            """)
            # Create index on timestamp for efficient date range queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_access_log_timestamp 
                ON access_log(timestamp DESC)
            """)
            # Create index on name for filtered queries
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_access_log_name 
                ON access_log(name)
            """)
            conn.commit()
            conn.close()
            logger.info(f"AccessLog database initialized: {self.log_path}")
        except sqlite3.Error as e:
            logger.error(f"Failed to initialize access log database: {e}")
    
    def _load_cache(self) -> None:
        """Load recent entries into memory cache for fast access."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT timestamp, name, access_granted, confidence 
                FROM access_log 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (self.max_memory_entries,))
            
            rows = cursor.fetchall()
            # Reverse to get chronological order (oldest first in cache)
            self._cache = [
                {
                    "timestamp": row["timestamp"],
                    "name": row["name"],
                    "access_granted": bool(row["access_granted"]),
                    "confidence": row["confidence"]
                }
                for row in reversed(rows)
            ]
            conn.close()
        except sqlite3.Error as e:
            logger.error(f"Failed to load access log cache: {e}")
            self._cache = []
    
    @property
    def entries(self) -> List[Dict[str, Any]]:
        """Property to access cache (for backward compatibility)."""
        return self._cache
    
    def load(self) -> None:
        """Reload cache from database (for backward compatibility)."""
        self._load_cache()
    
    def save(self) -> None:
        """No-op for backward compatibility - SQLite writes are immediate."""
        pass
    
    def add_entry(self, name: str, access_granted: bool, confidence: float = 0.0) -> Dict[str, Any]:
        """
        Record a new access event atomically to SQLite.
        
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
        
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO access_log (timestamp, name, access_granted, confidence)
                VALUES (?, ?, ?, ?)
            """, (entry["timestamp"], entry["name"], int(entry["access_granted"]), entry["confidence"]))
            conn.commit()
            conn.close()
            
            # Update cache
            self._cache.append(entry)
            if len(self._cache) > self.max_memory_entries:
                self._cache = self._cache[-self.max_memory_entries:]
                
        except sqlite3.Error as e:
            logger.error(f"Failed to add access log entry: {e}")
        
        return entry
    
    def get_recent(self, count: int = 50) -> List[Dict[str, Any]]:
        """
        Get the most recent N log entries in reverse chronological order.
        
        Args:
            count: Number of entries to retrieve
            
        Returns:
            List of entries, newest first
        """
        # Use cache if sufficient
        if count <= len(self._cache):
            return list(reversed(self._cache[-count:]))
        
        # Query database for larger requests
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                SELECT timestamp, name, access_granted, confidence 
                FROM access_log 
                ORDER BY timestamp DESC 
                LIMIT ?
            """, (count,))
            
            rows = cursor.fetchall()
            conn.close()
            
            return [
                {
                    "timestamp": row["timestamp"],
                    "name": row["name"],
                    "access_granted": bool(row["access_granted"]),
                    "confidence": row["confidence"]
                }
                for row in rows
            ]
        except sqlite3.Error as e:
            logger.error(f"Failed to get recent entries: {e}")
            return list(reversed(self._cache[-count:]))
    
    def get_paginated(self, page: int = 0, page_size: int = 50, 
                      date_from: Optional[Any] = None, date_to: Optional[Any] = None, 
                      name_filter: Optional[str] = None) -> Tuple[List[Dict[str, Any]], int, int]:
        """
        Get a page of log entries with optional filters using SQL.
        
        Args:
            page: Zero-indexed page number
            page_size: Number of entries per page
            date_from: Earliest date to include (inclusive)
            date_to: Latest date to include (inclusive)
            name_filter: Partial name match (case-insensitive)
            
        Returns:
            Tuple of (entries_list, total_matching_count, total_pages)
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Build WHERE clause dynamically
            conditions = []
            params: List[Any] = []
            
            if date_from:
                conditions.append("date(timestamp) >= ?")
                params.append(date_from.isoformat() if hasattr(date_from, 'isoformat') else str(date_from))
            
            if date_to:
                conditions.append("date(timestamp) <= ?")
                params.append(date_to.isoformat() if hasattr(date_to, 'isoformat') else str(date_to))
            
            if name_filter:
                conditions.append("name LIKE ?")
                params.append(f"%{name_filter}%")
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            # Get total count
            cursor.execute(f"SELECT COUNT(*) FROM access_log WHERE {where_clause}", params)
            total_count = cursor.fetchone()[0]
            total_pages = max(1, (total_count + page_size - 1) // page_size)
            
            # Get page data
            offset = page * page_size
            cursor.execute(f"""
                SELECT timestamp, name, access_granted, confidence 
                FROM access_log 
                WHERE {where_clause}
                ORDER BY timestamp DESC 
                LIMIT ? OFFSET ?
            """, params + [page_size, offset])
            
            rows = cursor.fetchall()
            conn.close()
            
            entries = [
                {
                    "timestamp": row["timestamp"],
                    "name": row["name"],
                    "access_granted": bool(row["access_granted"]),
                    "confidence": row["confidence"]
                }
                for row in rows
            ]
            
            return entries, total_count, total_pages
            
        except sqlite3.Error as e:
            logger.error(f"Failed to get paginated entries: {e}")
            return [], 0, 1
    
    def get_filtered(self, date_from: Optional[Any] = None, date_to: Optional[Any] = None, 
                     name_filter: Optional[str] = None, count: int = 100) -> List[Dict[str, Any]]:
        """
        Query log entries with optional filters using SQL.
        
        Args:
            date_from: Earliest date to include (inclusive)
            date_to: Latest date to include (inclusive)
            name_filter: Partial name match (case-insensitive)
            count: Maximum entries to return
            
        Returns:
            List of matching entries in reverse chronological order
        """
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Build WHERE clause dynamically
            conditions = []
            params: List[Any] = []
            
            if date_from:
                conditions.append("date(timestamp) >= ?")
                params.append(date_from.isoformat() if hasattr(date_from, 'isoformat') else str(date_from))
            
            if date_to:
                conditions.append("date(timestamp) <= ?")
                params.append(date_to.isoformat() if hasattr(date_to, 'isoformat') else str(date_to))
            
            if name_filter:
                conditions.append("name LIKE ?")
                params.append(f"%{name_filter}%")
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            cursor.execute(f"""
                SELECT timestamp, name, access_granted, confidence 
                FROM access_log 
                WHERE {where_clause}
                ORDER BY timestamp DESC 
                LIMIT ?
            """, params + [count])
            
            rows = cursor.fetchall()
            conn.close()
            
            return [
                {
                    "timestamp": row["timestamp"],
                    "name": row["name"],
                    "access_granted": bool(row["access_granted"]),
                    "confidence": row["confidence"]
                }
                for row in rows
            ]
            
        except sqlite3.Error as e:
            logger.error(f"Failed to get filtered entries: {e}")
            return []
    
    def get_unique_names(self) -> List[str]:
        """Get sorted list of all unique names in the access log."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT DISTINCT name FROM access_log ORDER BY name")
            names = [row[0] for row in cursor.fetchall()]
            conn.close()
            return names
        except sqlite3.Error as e:
            logger.error(f"Failed to get unique names: {e}")
            return sorted(set(entry['name'] for entry in self._cache))
    
    def get_total_count(self) -> int:
        """Get total number of log entries in database."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM access_log")
            count = cursor.fetchone()[0]
            conn.close()
            return count
        except sqlite3.Error as e:
            logger.error(f"Failed to get total count: {e}")
            return len(self._cache)
    
    def clear(self) -> None:
        """Delete all log entries from database."""
        try:
            conn = self._get_connection()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM access_log")
            conn.commit()
            # Reset auto-increment counter
            cursor.execute("DELETE FROM sqlite_sequence WHERE name='access_log'")
            conn.commit()
            conn.close()
            self._cache = []
            logger.info("Access log cleared")
        except sqlite3.Error as e:
            logger.error(f"Failed to clear access log: {e}")
    
    def migrate_from_json(self, json_path: str) -> int:
        """
        Import entries from an existing JSON log file.
        
        Args:
            json_path: Path to the JSON log file to migrate
            
        Returns:
            Number of entries imported
        """
        if not os.path.exists(json_path):
            logger.warning(f"JSON log file not found: {json_path}")
            return 0
        
        try:
            with open(json_path, 'r') as f:
                json_entries = json.load(f)
            
            if not json_entries:
                return 0
            
            conn = self._get_connection()
            cursor = conn.cursor()
            
            # Get existing timestamps to avoid duplicates
            cursor.execute("SELECT timestamp FROM access_log")
            existing = {row[0] for row in cursor.fetchall()}
            
            imported = 0
            for entry in json_entries:
                if entry['timestamp'] not in existing:
                    cursor.execute("""
                        INSERT INTO access_log (timestamp, name, access_granted, confidence)
                        VALUES (?, ?, ?, ?)
                    """, (
                        entry['timestamp'],
                        entry['name'],
                        int(entry['access_granted']),
                        entry.get('confidence', 0.0)
                    ))
                    imported += 1
            
            conn.commit()
            conn.close()
            
            # Reload cache
            self._load_cache()
            
            logger.info(f"Migrated {imported} entries from JSON to SQLite")
            return imported
            
        except (json.JSONDecodeError, IOError, sqlite3.Error) as e:
            logger.error(f"Failed to migrate from JSON: {e}")
            return 0


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
        Includes small sleep to reduce CPU spinning.
        """
        while self.is_running:
            try:
                if self.use_picamera:
                    frame = self.camera.capture_array()
                    frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                else:
                    ret, frame = self.camera.read()
                    if not ret:
                        time.sleep(0.01)  # Brief sleep on failed read
                        continue
                
                # Thread-safe frame update
                with self.frame_lock:
                    self.current_frame = frame
                
                # Small sleep to reduce CPU usage (configurable)
                time.sleep(Config.CAPTURE_THREAD_SLEEP_MS / 1000.0)
                    
            except cv2.error as e:
                logger.warning(f"OpenCV camera error: {e}")
                time.sleep(0.1)
            except OSError as e:
                logger.error(f"Camera OS error: {e}", exc_info=True)
                time.sleep(0.5)
    
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


# ==================== EMBEDDING MATCHING WORKER (MULTIPROCESSING) ====================
def _embedding_worker_process(
    input_queue,
    output_queue,
    stop_event,
    encodings_path: str,
    threshold: float,
    faiss_threshold: int
):
    """
    Standalone worker process for embedding similarity matching.
    
    Runs in a separate process to bypass GIL, enabling true parallel computation.
    Receives face embeddings from the main process and returns match results.
    
    Args:
        input_queue: Queue receiving (request_id, embeddings_list) tuples
        output_queue: Queue sending (request_id, results_list) tuples
        stop_event: Event to signal worker shutdown
        encodings_path: Path to encodings pickle file
        threshold: Recognition similarity threshold
        faiss_threshold: User count threshold for FAISS index
    """
    import numpy as np
    import pickle
    import os
    
    # Try to import FAISS in the worker process
    faiss_available = False
    faiss_index = None
    try:
        import faiss
        faiss_available = True
    except ImportError:
        pass
    
    # Load encodings in worker process
    known_encodings_normalized = None
    known_names = []
    
    def load_encodings():
        nonlocal known_encodings_normalized, known_names, faiss_index
        if os.path.exists(encodings_path):
            try:
                with open(encodings_path, "rb") as f:
                    data = pickle.loads(f.read())
                known_encodings = [np.array(enc, dtype=np.float32) for enc in data["encodings"]]
                known_names = data["names"]
                
                if len(known_encodings) > 0:
                    encodings_matrix = np.array(known_encodings, dtype=np.float32)
                    norms = np.linalg.norm(encodings_matrix, axis=1, keepdims=True)
                    known_encodings_normalized = (encodings_matrix / norms).astype(np.float32)
                    
                    # Build FAISS index if appropriate
                    if faiss_available and len(set(known_names)) >= faiss_threshold:
                        d = known_encodings_normalized.shape[1]
                        faiss_index = faiss.IndexFlatIP(d)
                        faiss_index.add(known_encodings_normalized)
                    else:
                        faiss_index = None
                else:
                    known_encodings_normalized = None
                    faiss_index = None
            except Exception:
                known_encodings_normalized = None
                faiss_index = None
    
    load_encodings()
    last_encodings_mtime = os.path.getmtime(encodings_path) if os.path.exists(encodings_path) else 0
    
    while not stop_event.is_set():
        try:
            # Check for encodings file update (reload if changed)
            if os.path.exists(encodings_path):
                current_mtime = os.path.getmtime(encodings_path)
                if current_mtime > last_encodings_mtime:
                    load_encodings()
                    last_encodings_mtime = current_mtime
            
            # Get work from queue with timeout
            try:
                request_id, embeddings = input_queue.get(timeout=0.1)
            except:
                continue
            
            results = []
            
            if known_encodings_normalized is None or len(embeddings) == 0:
                output_queue.put((request_id, results))
                continue
            
            for face_data in embeddings:
                face_encoding = np.array(face_data['embedding'], dtype=np.float32)
                face_norm = face_encoding / np.linalg.norm(face_encoding)
                
                # Use FAISS or linear search
                if faiss_index is not None:
                    query = face_norm.reshape(1, -1).astype(np.float32)
                    similarities, indices = faiss_index.search(query, 1)
                    best_match_index = int(indices[0][0])
                    best_similarity = float(similarities[0][0])
                else:
                    similarities = np.dot(known_encodings_normalized, face_norm)
                    best_match_index = np.argmax(similarities)
                    best_similarity = float(similarities[best_match_index])
                
                if best_similarity > threshold:
                    name = known_names[best_match_index]
                    confidence = best_similarity
                else:
                    name = "Unknown"
                    confidence = 0.0
                
                results.append({
                    'name': name,
                    'confidence': confidence,
                    'location': face_data['location'],
                    'is_stable': face_data['is_stable']
                })
            
            output_queue.put((request_id, results))
            
        except Exception as e:
            # Don't crash the worker on errors
            continue


# ==================== BLINK LIVENESS DETECTOR ====================
class BlinkDetector:
    """
    Liveness detection using Eye Aspect Ratio (EAR) blink detection.
    
    Uses MediaPipe Face Mesh to detect eye landmarks and calculate EAR.
    A blink is detected when EAR drops below threshold for consecutive frames.
    This prevents photo/video spoofing attacks since printed faces can't blink.
    """
    
    # MediaPipe Face Mesh eye landmark indices
    # Left eye landmarks (from user's perspective, so right side of image)
    LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
    # Right eye landmarks
    RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
    
    def __init__(self):
        """Initialize the blink detector with MediaPipe Face Mesh."""
        self.available = False
        self.face_mesh = None
        
        # Per-face tracking state: {face_id: {'blink_count': int, 'consec_frames': int, 'start_time': float, 'passed': bool}}
        self._tracking: Dict[str, Dict[str, Any]] = {}
        
        if not USE_MEDIAPIPE:
            logger.warning("MediaPipe not available - blink detection disabled")
            return
        
        try:
            mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=5,
                refine_landmarks=True,  # Includes iris landmarks for better eye detection
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.available = True
            logger.info("Blink detector initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize blink detector: {e}")
    
    def _calculate_ear(self, eye_landmarks: List[Tuple[float, float]]) -> float:
        """
        Calculate Eye Aspect Ratio (EAR) for a single eye.
        
        EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
        
        Where p1-p6 are the 6 eye landmarks in order:
        p1: outer corner, p2: upper outer, p3: upper inner,
        p4: inner corner, p5: lower inner, p6: lower outer
        
        Args:
            eye_landmarks: List of 6 (x, y) tuples for eye landmarks
            
        Returns:
            Eye Aspect Ratio (higher = more open, lower = more closed)
        """
        # Vertical distances
        v1 = np.linalg.norm(np.array(eye_landmarks[1]) - np.array(eye_landmarks[5]))
        v2 = np.linalg.norm(np.array(eye_landmarks[2]) - np.array(eye_landmarks[4]))
        
        # Horizontal distance
        h = np.linalg.norm(np.array(eye_landmarks[0]) - np.array(eye_landmarks[3]))
        
        if h == 0:
            return 0.0
        
        ear = (v1 + v2) / (2.0 * h)
        return ear
    
    def _get_face_id(self, bbox: Tuple[int, int, int, int], name: Optional[str] = None) -> str:
        """
        Generate a stable ID for a face.
        
        Uses the recognized name if available (most stable), otherwise falls back
        to position-based ID with larger bins to reduce flickering.
        """
        if name and name not in ("Unknown", "Scanning..."):
            return f"name_{name}"
        
        left, top, right, bottom = bbox
        # Use center position binned with larger bins (100px) for more stability
        cx = (left + right) // 2 // 100 * 100
        cy = (top + bottom) // 2 // 100 * 100
        return f"pos_{cx}_{cy}"
    
    def check_blink(self, frame: np.ndarray, bbox: Tuple[int, int, int, int], 
                    name: Optional[str] = None) -> Tuple[bool, int, float, bool]:
        """
        Check for blink in the given face region.
        
        Args:
            frame: Full BGR frame
            bbox: Face bounding box (left, top, right, bottom)
            name: Recognized person name (used for stable tracking)
            
        Returns:
            Tuple of (liveness_passed, blink_count, current_ear, awaiting_blink)
            - liveness_passed: True if required blinks detected within timeout
            - blink_count: Number of blinks detected so far
            - current_ear: Current Eye Aspect Ratio (for debugging/display)
            - awaiting_blink: True if actively waiting for user to blink
        """
        if not self.available or not Config.ENABLE_BLINK_DETECTION:
            return True, 0, 1.0, False, False  # Pass if detection disabled
        
        face_id = self._get_face_id(bbox, name)
        now = time.time()
        
        # Initialize tracking for new face
        if face_id not in self._tracking:
            self._tracking[face_id] = {
                'blink_count': 0,
                'consec_frames': 0,
                'start_time': now,
                'last_seen': now,
                'passed': False,
                'last_ear': 1.0,
                'awaiting_blink': True  # Start in awaiting state
            }
        
        state = self._tracking[face_id]
        state['last_seen'] = now  # Update last seen time
        
        # Note: We no longer cache 'passed' state here - blink verification is tracked
        # at the app level in self.blink_verified to ensure each access attempt 
        # requires a fresh blink verification
        
        # Check timeout for blink detection window
        if now - state['start_time'] > Config.BLINK_TIMEOUT_SECONDS:
            # Reset and try again
            state['blink_count'] = 0
            state['consec_frames'] = 0
            state['start_time'] = now
            state['awaiting_blink'] = True
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            results = self.face_mesh.process(rgb_frame)
            
            if not results.multi_face_landmarks:
                return False, state['blink_count'], state['last_ear'], state['awaiting_blink']
            
            # Find the face mesh that best matches our bbox
            h, w = frame.shape[:2]
            best_landmarks = None
            best_overlap = 0
            
            for face_landmarks in results.multi_face_landmarks:
                # Get bounding box of this face mesh
                xs = [lm.x * w for lm in face_landmarks.landmark]
                ys = [lm.y * h for lm in face_landmarks.landmark]
                mesh_left, mesh_right = int(min(xs)), int(max(xs))
                mesh_top, mesh_bottom = int(min(ys)), int(max(ys))
                
                # Calculate overlap with our bbox
                left, top, right, bottom = bbox
                overlap_left = max(left, mesh_left)
                overlap_right = min(right, mesh_right)
                overlap_top = max(top, mesh_top)
                overlap_bottom = min(bottom, mesh_bottom)
                
                if overlap_right > overlap_left and overlap_bottom > overlap_top:
                    overlap_area = (overlap_right - overlap_left) * (overlap_bottom - overlap_top)
                    if overlap_area > best_overlap:
                        best_overlap = overlap_area
                        best_landmarks = face_landmarks
            
            if best_landmarks is None:
                return False, state['blink_count'], state['last_ear'], state['awaiting_blink']
            
            # Extract eye landmarks
            left_eye = [(best_landmarks.landmark[i].x * w, best_landmarks.landmark[i].y * h) 
                        for i in self.LEFT_EYE_INDICES]
            right_eye = [(best_landmarks.landmark[i].x * w, best_landmarks.landmark[i].y * h) 
                         for i in self.RIGHT_EYE_INDICES]
            
            # Calculate EAR for both eyes
            left_ear = self._calculate_ear(left_eye)
            right_ear = self._calculate_ear(right_eye)
            avg_ear = (left_ear + right_ear) / 2.0
            state['last_ear'] = avg_ear
            
            # Check for blink (EAR below threshold)
            if avg_ear < Config.EAR_THRESHOLD:
                state['consec_frames'] += 1
            else:
                # If we had enough consecutive low-EAR frames, count it as a blink
                if state['consec_frames'] >= Config.BLINK_CONSEC_FRAMES:
                    state['blink_count'] += 1
                    logger.info(f"Blink detected for {face_id}, count: {state['blink_count']}")
                state['consec_frames'] = 0
            
            # Check if liveness passed
            if state['blink_count'] >= Config.BLINK_REQUIRED_COUNT:
                # Don't cache 'passed' state - let the app level handle it
                state['awaiting_blink'] = False
                logger.info(f"Liveness verified for {face_id} ({state['blink_count']} blinks)")
                return True, state['blink_count'], avg_ear, False
            
            return False, state['blink_count'], avg_ear, state['awaiting_blink']
            
        except Exception as e:
            logger.error(f"Blink detection error: {e}")
            return False, state['blink_count'], state['last_ear'], state['awaiting_blink']
    
    def reset_face(self, bbox: Tuple[int, int, int, int]) -> None:
        """Reset tracking state for a specific face."""
        face_id = self._get_face_id(bbox)
        if face_id in self._tracking:
            del self._tracking[face_id]
    
    def cleanup_stale(self, max_age_seconds: float = 10.0) -> None:
        """Remove tracking state for faces not seen recently."""
        now = time.time()
        stale_ids = [
            fid for fid, state in self._tracking.items()
            if now - state.get('last_seen', state['start_time']) > max_age_seconds
        ]
        for fid in stale_ids:
            del self._tracking[fid]
    
    def close(self) -> None:
        """Release MediaPipe resources."""
        if self.face_mesh:
            self.face_mesh.close()
            self.face_mesh = None
            self.available = False


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
        self.disabled_encodings_path = Config.DISABLED_ENCODINGS_PATH
        self.known_encodings = []
        self.known_names = []
        self.known_encodings_normalized = None  # Pre-computed for fast matching
        self.faiss_index = None  # FAISS index for fast vector search (built when user count is high)
        self.disabled_encodings = {}  # {name: [encodings]} for revoked users
        self.cv_scaler = Config.DETECTION_SCALE_FACTOR
        
        # Face stability tracking - wait for face to settle before recognizing
        self.stability_tracker = FaceStabilityTracker()
        
        # Initialize InsightFace model
        try:
            self.face_app = FaceAnalysis(
                name='buffalo_s',  # Lightweight model suitable for edge devices
                providers=ONNX_PROVIDERS
            )
            self.face_app.prepare(ctx_id=0, det_size=(640, 640))
            logger.info(f"InsightFace model (buffalo_s) loaded with providers: {ONNX_PROVIDERS}")
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
        
        # Blink-based liveness detection
        self.blink_detector = None
        if Config.ENABLE_BLINK_DETECTION:
            self.blink_detector = BlinkDetector()
            if self.blink_detector.available:
                logger.info("Blink-based liveness detection enabled")
            else:
                logger.warning("Blink detection requested but MediaPipe not available")
        
        # Frame skip counter for performance
        self.frame_count = 0
        
        # ========== THREADED/MULTIPROCESS RECOGNITION ==========
        # Background thread/process prevents recognition from blocking the camera loop
        self._recognition_thread = None
        self._recognition_lock = threading.Lock()
        self._frame_queue = Queue(maxsize=2)  # Limit queue size to prevent memory buildup
        self._last_results = []  # Most recent recognition results
        self._last_results_lock = threading.Lock()
        self._recognition_running = False
        self._stop_recognition = threading.Event()
        
        # Multiprocessing components (used if USE_MULTIPROCESSING is True)
        self._use_multiprocessing = Config.USE_MULTIPROCESSING
        self._mp_worker = None
        self._mp_input_queue = None
        self._mp_output_queue = None
        self._mp_stop_event = None
        self._mp_request_id = 0
        self._mp_pending_requests = {}
        
        # Ensure dataset directory exists
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
        
        self.load_encodings()
        self.load_disabled_encodings()
    
    def load_disabled_encodings(self):
        """Load revoked user encodings from disk."""
        if os.path.exists(self.disabled_encodings_path):
            try:
                with open(self.disabled_encodings_path, "rb") as f:
                    self.disabled_encodings = pickle.loads(f.read())
                logger.info(f"Loaded {len(self.disabled_encodings)} disabled users")
            except (OSError, IOError, pickle.UnpicklingError, KeyError) as e:
                logger.warning(f"Could not load disabled encodings: {e}", exc_info=True)
                self.disabled_encodings = {}
        else:
            self.disabled_encodings = {}
    
    def save_disabled_encodings(self):
        """Save revoked user encodings to disk."""
        try:
            with open(self.disabled_encodings_path, "wb") as f:
                f.write(pickle.dumps(self.disabled_encodings))
        except IOError as e:
            logger.error(f"Failed to save disabled encodings: {e}")
    
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
            except (OSError, IOError, pickle.UnpicklingError, KeyError) as e:
                logger.error(f"Failed to load encodings: {e}", exc_info=True)
                return False
        return False
    
    def _update_normalized_encodings(self) -> None:
        """
        Pre-compute normalized encoding matrix for fast similarity.
        Normalizing once here avoids per-comparison normalization overhead.
        Also builds FAISS index if available and user count exceeds threshold.
        """
        if len(self.known_encodings) > 0:
            encodings_matrix = np.array(self.known_encodings, dtype=np.float32)
            norms = np.linalg.norm(encodings_matrix, axis=1, keepdims=True)
            self.known_encodings_normalized = (encodings_matrix / norms).astype(np.float32)
            
            # Build FAISS index for fast similarity search with large user counts
            self._build_faiss_index()
        else:
            self.known_encodings_normalized = None
            self.faiss_index = None
    
    def _build_faiss_index(self) -> None:
        """
        Build FAISS index for fast vector similarity search.
        Uses Inner Product (IP) index since embeddings are pre-normalized (IP = cosine similarity).
        Falls back to linear search if FAISS unavailable or user count is below threshold.
        """
        if not USE_FAISS:
            self.faiss_index = None
            return
        
        num_users = len(set(self.known_names))
        if num_users < Config.FAISS_INDEX_THRESHOLD:
            self.faiss_index = None
            logger.debug(f"FAISS index not built: {num_users} users below threshold {Config.FAISS_INDEX_THRESHOLD}")
            return
        
        try:
            # Embedding dimension (InsightFace buffalo_s uses 512-dim embeddings)
            d = self.known_encodings_normalized.shape[1]
            
            # Use IndexFlatIP for exact inner product (cosine similarity for normalized vectors)
            self.faiss_index = faiss.IndexFlatIP(d)
            self.faiss_index.add(self.known_encodings_normalized)
            
            logger.info(f"Built FAISS index with {len(self.known_encodings)} encodings (dim={d})")
        except Exception as e:
            logger.warning(f"Failed to build FAISS index, falling back to linear search: {e}")
            self.faiss_index = None
    
    def _search_faiss(self, face_encoding: np.ndarray) -> Tuple[int, float]:
        """
        Search FAISS index for best matching face.
        
        Args:
            face_encoding: Normalized face embedding vector
            
        Returns:
            Tuple of (best_match_index, similarity_score)
        """
        # Reshape for FAISS query (expects 2D array)
        query = face_encoding.reshape(1, -1).astype(np.float32)
        
        # Search for top-1 match
        similarities, indices = self.faiss_index.search(query, 1)
        
        return int(indices[0][0]), float(similarities[0][0])
    
    def get_registered_persons(self) -> List[Tuple[str, int]]:
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
    
    def get_trained_persons(self) -> List[str]:
        """Get list of unique names that have been trained (have encodings)."""
        return list(set(self.known_names))
    
    def create_person_folder(self, name: str) -> str:
        """Create a directory for storing a person's face images."""
        person_folder = os.path.join(self.dataset_path, name)
        if not os.path.exists(person_folder):
            os.makedirs(person_folder)
        return person_folder
    
    def save_face_image(self, frame: np.ndarray, name: str) -> str:
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
    
    def train_model(self, progress_callback: Optional[Callable[[int, int, str], None]] = None) -> Tuple[bool, str]:
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
    
    def remove_person_from_model(self, person_name: str) -> Tuple[bool, str]:
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
    
    def revoke_person_access(self, person_name: str) -> Tuple[bool, str]:
        """
        Revoke access for a person by moving their encodings to disabled list.
        Encodings are preserved and can be restored later without retraining.
        
        Args:
            person_name: Name of person to revoke
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        if not self.known_names:
            return False, "No trained model exists"
        
        if person_name not in self.known_names:
            return False, f"{person_name} not found in model"
        
        # Extract encodings for this person
        person_encodings = [self.known_encodings[i] for i, name in enumerate(self.known_names) if name == person_name]
        
        # Store in disabled list
        self.disabled_encodings[person_name] = [enc.tolist() if hasattr(enc, 'tolist') else enc for enc in person_encodings]
        self.save_disabled_encodings()
        
        # Remove from active model
        indices_to_keep = [i for i, name in enumerate(self.known_names) if name != person_name]
        self.known_encodings = [self.known_encodings[i] for i in indices_to_keep]
        self.known_names = [self.known_names[i] for i in indices_to_keep]
        
        # Backup and save
        if Config.BACKUP_ENABLED:
            BackupManager.create_backup(self.encodings_path, Config.MAX_BACKUPS)
        
        if self.known_encodings:
            data = {"encodings": [enc.tolist() if hasattr(enc, 'tolist') else enc for enc in self.known_encodings], "names": self.known_names}
            with open(self.encodings_path, "wb") as f:
                f.write(pickle.dumps(data))
            self._update_normalized_encodings()
        else:
            if os.path.exists(self.encodings_path):
                os.remove(self.encodings_path)
            self.known_encodings_normalized = None
        
        self.face_cache.clear()
        
        return True, f"Revoked access for {person_name} ({len(person_encodings)} encodings preserved)"
    
    def restore_person_access(self, person_name: str) -> Tuple[bool, str]:
        """
        Restore access for a previously revoked person.
        Moves their encodings back from disabled list to active model.
        
        Args:
            person_name: Name of person to restore
            
        Returns:
            Tuple of (success: bool, message: str)
        """
        if person_name not in self.disabled_encodings:
            return False, f"{person_name} not found in revoked users"
        
        # Retrieve stored encodings
        person_encodings = [np.array(enc, dtype=np.float32) for enc in self.disabled_encodings[person_name]]
        count = len(person_encodings)
        
        # Add back to active model
        self.known_encodings.extend(person_encodings)
        self.known_names.extend([person_name] * count)
        
        # Remove from disabled list
        del self.disabled_encodings[person_name]
        self.save_disabled_encodings()
        
        # Backup and save active model
        if Config.BACKUP_ENABLED:
            BackupManager.create_backup(self.encodings_path, Config.MAX_BACKUPS)
        
        data = {"encodings": [enc.tolist() if hasattr(enc, 'tolist') else enc for enc in self.known_encodings], "names": self.known_names}
        with open(self.encodings_path, "wb") as f:
            f.write(pickle.dumps(data))
        
        self._update_normalized_encodings()
        self.face_cache.clear()
        
        return True, f"Restored access for {person_name} ({count} encodings)"
    
    def get_disabled_persons(self):
        """Get list of revoked users with their encoding counts."""
        return [(name, len(encodings)) for name, encodings in self.disabled_encodings.items()]
    
    def detect_faces_fast(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
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
    
    def detect_faces_robust(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
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
    
    def detect_faces_combined(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Hybrid detection: fast Haar cascade first, then InsightFace fallback.
        Optimizes for speed while maintaining detection reliability.
        """
        faces = self.detect_faces_fast(frame)
        
        # Fall back to robust detection if fast detection finds nothing
        if not faces:
            faces = self.detect_faces_robust(frame)
        
        return faces
    
    # ========== BACKGROUND RECOGNITION THREAD/PROCESS METHODS ==========
    
    def start_recognition_thread(self) -> None:
        """Launch the background recognition processing thread/process."""
        if self._recognition_thread is not None and self._recognition_thread.is_alive():
            return
        
        # Start multiprocessing worker if enabled
        if self._use_multiprocessing:
            self._start_mp_worker()
        
        self._stop_recognition.clear()
        self._recognition_thread = threading.Thread(target=self._recognition_loop, daemon=True)
        self._recognition_thread.start()
        logger.info(f"Background recognition started (multiprocessing={self._use_multiprocessing})")
    
    def _start_mp_worker(self) -> None:
        """Start the multiprocessing worker for embedding matching."""
        if self._mp_worker is not None and self._mp_worker.is_alive():
            return
        
        self._mp_input_queue = MPQueue(maxsize=5)
        self._mp_output_queue = MPQueue(maxsize=5)
        self._mp_stop_event = MPEvent()
        
        self._mp_worker = Process(
            target=_embedding_worker_process,
            args=(
                self._mp_input_queue,
                self._mp_output_queue,
                self._mp_stop_event,
                self.encodings_path,
                Config.RECOGNITION_THRESHOLD,
                Config.FAISS_INDEX_THRESHOLD
            ),
            daemon=True
        )
        self._mp_worker.start()
        logger.info("Multiprocessing embedding worker started")
    
    def _stop_mp_worker(self) -> None:
        """Stop the multiprocessing worker."""
        if self._mp_stop_event is not None:
            self._mp_stop_event.set()
        
        if self._mp_worker is not None:
            self._mp_worker.join(timeout=2.0)
            if self._mp_worker.is_alive():
                self._mp_worker.terminate()
            self._mp_worker = None
        
        # Clean up queues
        for q in [self._mp_input_queue, self._mp_output_queue]:
            if q is not None:
                try:
                    while not q.empty():
                        q.get_nowait()
                except:
                    pass
        
        self._mp_input_queue = None
        self._mp_output_queue = None
        self._mp_stop_event = None
        logger.info("Multiprocessing embedding worker stopped")
    
    def stop_recognition_thread(self) -> None:
        """Gracefully shut down the recognition thread/process."""
        self._stop_recognition.set()
        
        # Stop multiprocessing worker
        if self._use_multiprocessing:
            self._stop_mp_worker()
        
        # Drain queue to unblock the worker thread
        try:
            while not self._frame_queue.empty():
                self._frame_queue.get_nowait()
        except Empty:
            pass
        
        if self._recognition_thread is not None:
            self._recognition_thread.join(timeout=2.0)
            self._recognition_thread = None
        logger.info("Background recognition stopped")
    
    def _recognition_loop(self) -> None:
        """
        Main loop for background recognition thread.
        Continuously pulls frames from queue and processes them.
        Uses multiprocessing for embedding matching if enabled.
        """
        while not self._stop_recognition.is_set():
            try:
                frame = self._frame_queue.get(timeout=0.1)
            except Empty:
                # Check for multiprocessing results even when no new frames
                if self._use_multiprocessing:
                    self._collect_mp_results()
                continue
            
            with self._recognition_lock:
                self._recognition_running = True
                try:
                    if self._use_multiprocessing:
                        results = self._process_recognition_mp(frame)
                    else:
                        results = self._process_recognition(frame)
                    with self._last_results_lock:
                        self._last_results = results
                except cv2.error as e:
                    logger.error(f"OpenCV recognition error: {e}")
                except (ValueError, TypeError) as e:
                    logger.error(f"Recognition data error: {e}", exc_info=True)
                except RuntimeError as e:
                    logger.error(f"Recognition runtime error: {e}", exc_info=True)
                finally:
                    self._recognition_running = False
                    # Help GC by clearing reference
                    del frame
    
    def _collect_mp_results(self) -> None:
        """Collect any pending results from the multiprocessing worker."""
        if self._mp_output_queue is None:
            return
        
        try:
            while not self._mp_output_queue.empty():
                request_id, results = self._mp_output_queue.get_nowait()
                # Store results keyed by request ID
                self._mp_pending_requests[request_id] = results
        except:
            pass
    
    def _process_recognition_mp(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Process recognition using multiprocessing worker for embedding matching.
        
        Face detection happens in this thread (using InsightFace),
        but the heavy embedding comparison is offloaded to a separate process.
        """
        # First, collect any pending results from worker
        self._collect_mp_results()
        
        results = []
        
        # Run InsightFace detection and embedding extraction (in this thread)
        faces = self.face_app.get(frame)
        
        embeddings_to_match = []
        face_metadata = []  # Store metadata for each face to correlate with results
        
        for face_idx, face in enumerate(faces):
            if face.embedding is None:
                continue
            
            face_encoding = face.embedding
            bbox = face.bbox.astype(int)
            left, top, right, bottom = bbox[0], bbox[1], bbox[2], bbox[3]
            location = (top, right, bottom, left)
            
            # Check face stability
            is_stable = self.stability_tracker.update_and_check_stability(face_idx, location)
            
            # Try cache lookup first
            cached = self.face_cache.get_by_embedding(face_encoding)
            if cached:
                self.face_cache.update_location(cached['location'], location)
                results.append({
                    'name': cached['name'],
                    'confidence': cached['confidence'],
                    'location': location,
                    'from_cache': True,
                    'is_stable': is_stable
                })
                continue
            
            # Check position-based cache
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
            
            # Queue for matching in worker process
            embeddings_to_match.append({
                'embedding': face_encoding.tolist(),
                'location': location,
                'is_stable': is_stable
            })
            face_metadata.append({
                'encoding': face_encoding,
                'location': location
            })
        
        # Send to worker process for matching
        if embeddings_to_match and self._mp_input_queue is not None:
            self._mp_request_id += 1
            try:
                self._mp_input_queue.put_nowait((self._mp_request_id, embeddings_to_match))
            except:
                pass  # Queue full, will retry next frame
            
            # Wait briefly for result (with timeout to stay responsive)
            start_wait = time.time()
            while time.time() - start_wait < 0.05:  # 50ms timeout
                self._collect_mp_results()
                if self._mp_request_id in self._mp_pending_requests:
                    mp_results = self._mp_pending_requests.pop(self._mp_request_id)
                    
                    # Add results and update cache
                    for i, mp_result in enumerate(mp_results):
                        if mp_result['name'] != "Unknown":
                            self.face_cache.put(
                                mp_result['location'],
                                mp_result['name'],
                                mp_result['confidence'],
                                face_metadata[i]['encoding']
                            )
                        results.append({
                            'name': mp_result['name'],
                            'confidence': mp_result['confidence'],
                            'location': mp_result['location'],
                            'from_cache': False,
                            'is_stable': mp_result['is_stable']
                        })
                    break
                time.sleep(0.005)
            else:
                # Timeout - add pending faces as "Scanning..."
                for meta in face_metadata:
                    results.append({
                        'name': 'Scanning...',
                        'confidence': 0.0,
                        'location': meta['location'],
                        'from_cache': False,
                        'is_stable': True
                    })
        
        return results
    
    def _process_recognition(self, frame: np.ndarray) -> List[Dict[str, Any]]:
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
            
            # Perform actual recognition using FAISS or linear search
            name = "Unknown"
            confidence = 0.0
            
            # Normalize current embedding
            face_norm = (face_encoding / np.linalg.norm(face_encoding)).astype(np.float32)
            
            # Use FAISS index if available, otherwise fall back to linear search
            if self.faiss_index is not None:
                best_match_index, best_similarity = self._search_faiss(face_norm)
            else:
                # Linear search with vectorized cosine similarity
                similarities = np.dot(self.known_encodings_normalized, face_norm)
                best_match_index = np.argmax(similarities)
                best_similarity = similarities[best_match_index]
            
            if best_similarity > Config.RECOGNITION_THRESHOLD:
                name = self.known_names[best_match_index]
                confidence = float(best_similarity)
            
            # Only cache recognized faces, not unknown ones
            if name != "Unknown":
                self.face_cache.put(location, name, confidence, face_encoding)
            
            results.append({
                'name': name,
                'confidence': confidence,
                'location': location,
                'from_cache': False,
                'is_stable': True
            })
        
        return results
    
    def recognize_faces(self, frame, force_recognition=False, idle_mode=False):
        """
        Non-blocking face recognition using background thread.
        
        Submits frame for processing and immediately returns cached/previous results.
        This prevents the camera loop from being blocked by slow recognition.
        
        Args:
            frame: OpenCV BGR image
            force_recognition: Skip frame interval check
            idle_mode: Use less frequent recognition when no faces detected
            
        Returns:
            Tuple of (frame, list of recognition results)
        """
        self.frame_count += 1
        
        if not self.known_encodings or self.known_encodings_normalized is None:
            return frame, []
        
        # Use adaptive recognition interval based on detection state
        recognition_interval = Config.IDLE_RECOGNITION_INTERVAL if idle_mode else Config.RECOGNITION_INTERVAL_FRAMES
        
        # Implement frame skipping for performance
        if not force_recognition and self.frame_count % recognition_interval != 0:
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
    
    def recognize_faces_sync(self, frame: np.ndarray, force_recognition: bool = False) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Synchronous face recognition (blocks until complete).
        Use for single-frame recognition scenarios.
        """
        if not self.known_encodings or self.known_encodings_normalized is None:
            return frame, []
        
        return frame, self._process_recognition(frame)
    
    def get_last_results(self) -> List[Dict[str, Any]]:
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
        
        # Auto-migrate from JSON if old log exists
        old_json_log = "access_log.json"
        if os.path.exists(old_json_log):
            migrated = self.access_log.migrate_from_json(old_json_log)
            if migrated > 0:
                # Rename old file to prevent re-migration
                backup_path = "access_log.json.migrated"
                try:
                    os.rename(old_json_log, backup_path)
                    logger.info(f"Old JSON log backed up to {backup_path}")
                except OSError as e:
                    logger.warning(f"Could not rename old JSON log: {e}")
        
        # ========== Application State ==========
        self.is_running = True
        self.is_scanning = True
        self.camera_thread = None
        self.current_status = "scanning"  # States: scanning, granted, denied
        self.status_message = ""
        self.status_set_time = 0  # Timestamp when status was set (to protect granted/denied)
        self.last_access = {}  # Cooldown tracking per person
        self.admin_mode = False
        
        # Blink detection mode - when active, skip face recognition to save resources
        self.awaiting_blink_mode = False
        self.awaiting_blink_name = None  # Name of person awaiting blink verification
        self.awaiting_blink_location = None  # Last known face location
        
        # Track who has verified blink for current access attempt
        # Key: name, Value: timestamp when blink was verified
        # This prevents granting access without blink verification
        self.blink_verified = {}  # {name: verification_timestamp}
        
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
        self.last_gc_run = time.time()
        self.cache_cleanup_interval = 60  # Cleanup every 60 seconds
        
        # ========== Adaptive Frame Rate State ==========
        self.target_fps = Config.IDLE_FPS  # Start in idle mode
        self.no_face_frames = 0  # Counter for consecutive frames without faces
        self.idle_threshold_frames = 15  # Switch to idle mode after this many empty frames
        
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
        self.status_set_time = time.time()  # Record when status was set
        
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
            
        elif status == "awaiting_blink":
            # Liveness check - waiting for blink
            self.status_icon_label.config(text="👁")
            self.status_text_label.config(text="Please Blink")
            self.status_detail_label.config(text="Liveness verification required")
            update_bg(Config.COLOR_CARD, Config.COLOR_SCANNING, Config.COLOR_TEXT, Config.COLOR_TEXT_SECONDARY)
            self.status_card.config(highlightbackground=Config.COLOR_SCANNING)
            
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
                    
                    elif self.awaiting_blink_mode and self.awaiting_blink_name:
                        # ========== BLINK-ONLY MODE ==========
                        # Skip expensive face recognition - only run blink detection
                        # This gives maximum resources to camera feed and EAR detection
                        
                        if self.face_system.blink_detector and self.face_system.blink_detector.available:
                            # Use lightweight face detection just to find face location
                            faces = self.face_system.detect_faces_combined(frame)
                            
                            if len(faces) > 0:
                                # Use first detected face
                                top, right, bottom, left = faces[0]
                                bbox = (left, top, right, bottom)
                                self.awaiting_blink_location = (top, right, bottom, left)
                                
                                # Run blink detection (lightweight compared to full recognition)
                                liveness_passed, blink_count, ear, awaiting = self.face_system.blink_detector.check_blink(
                                    frame, bbox, self.awaiting_blink_name
                                )
                                
                                if liveness_passed:
                                    # Blink detected! Record verification and grant access
                                    now = time.time()
                                    name = self.awaiting_blink_name
                                    
                                    # Record blink verification timestamp
                                    self.blink_verified[name] = now
                                    
                                    # Exit blink mode
                                    self.awaiting_blink_mode = False
                                    self.awaiting_blink_name = None
                                    self.awaiting_blink_location = None
                                    
                                    # Reset blink detector state for this person so next time requires fresh blink
                                    if self.face_system.blink_detector:
                                        face_id = f"name_{name}"
                                        if face_id in self.face_system.blink_detector._tracking:
                                            del self.face_system.blink_detector._tracking[face_id]
                                    
                                    if name not in self.last_access or (now - self.last_access[name]) > Config.COOLDOWN_SECONDS:
                                        self.last_access[name] = now
                                        # Use high confidence since they were already recognized
                                        self.root.after(0, lambda n=name: self.grant_access(n, 0.95))
                                # else: keep waiting for blink, status already shows "Please Blink"
                            else:
                                # Face disappeared - exit blink mode and return to scanning
                                self.awaiting_blink_mode = False
                                self.awaiting_blink_name = None
                                self.awaiting_blink_location = None
                                self.root.after(0, self._set_status_scanning)
                        
                        # Clean up stale blink verifications (older than 5 seconds)
                        now = time.time()
                        stale_blinks = [n for n, t in self.blink_verified.items() if now - t > 5.0]
                        for n in stale_blinks:
                            del self.blink_verified[n]
                    
                    else:
                        # ========== NORMAL RECOGNITION MODE ==========
                        # Run face recognition with adaptive interval based on idle state
                        is_idle = self.no_face_frames >= self.idle_threshold_frames
                        _, results = self.face_system.recognize_faces(frame, idle_mode=is_idle)
                        self.faces_detected = len(results)
                        
                        # Track if any face is awaiting blink verification
                        any_awaiting_blink = False
                        pending_blink_name = None
                        pending_blink_location = None
                        
                        # First pass: check blink status for all recognized faces
                        for result in results:
                            name = result['name']
                            confidence = result['confidence']
                            is_stable = result.get('is_stable', True)
                            location = result.get('location', (0, 0, 0, 0))
                            
                            if not is_stable or name == 'Scanning...' or name == "Unknown":
                                continue
                            
                            if name != "Unknown" and confidence >= Config.RECOGNITION_THRESHOLD:
                                # Skip blink check if person is within cooldown - they already passed
                                now = time.time()
                                if name in self.last_access and (now - self.last_access[name]) <= Config.COOLDOWN_SECONDS:
                                    continue  # Still in cooldown, don't request blink
                                
                                if self.face_system.blink_detector and self.face_system.blink_detector.available:
                                    top, right, bottom, left = location
                                    bbox = (left, top, right, bottom)
                                    liveness_passed, _, _, awaiting = self.face_system.blink_detector.check_blink(frame, bbox, name)
                                    if not liveness_passed and awaiting:
                                        any_awaiting_blink = True
                                        pending_blink_name = name
                                        pending_blink_location = location
                        
                        # If awaiting blink, enter blink-only mode to save resources
                        if any_awaiting_blink and pending_blink_name:
                            self.awaiting_blink_mode = True
                            self.awaiting_blink_name = pending_blink_name
                            self.awaiting_blink_location = pending_blink_location
                            if self.current_status != "awaiting_blink":
                                self.root.after(0, lambda: self.set_status("awaiting_blink"))
                            # Skip rest of processing - will handle in blink-only mode next frame
                            pass
                        else:
                            # Check face stability status
                            any_unstable = any(not r.get('is_stable', True) for r in results)
                            any_stable = any(r.get('is_stable', True) for r in results)
                            
                            # Protect granted/denied status from being overwritten too quickly
                            status_age = time.time() - self.status_set_time
                            status_protected = self.current_status in ("granted", "denied") and status_age < 2.0
                            
                            # Update UI status based on face detection (only if not protected)
                            if not status_protected:
                                if len(results) > 0 and self.current_status not in ("granted", "denied"):
                                    if any_unstable and not any_stable:
                                        self.root.after(0, lambda: self.set_status("processing"))
                                    else:
                                        self.root.after(0, self._set_status_active_scanning)
                                elif len(results) == 0 and self.current_status in ("active_scanning", "processing", "awaiting_blink"):
                                    self.root.after(0, self._set_status_scanning)
                        
                        # Process each recognized face (only if not in blink-waiting mode)
                        if not any_awaiting_blink:
                            for result in results:
                                name = result['name']
                                confidence = result['confidence']
                                from_cache = result.get('from_cache', False)
                                is_stable = result.get('is_stable', True)
                                location = result.get('location', (0, 0, 0, 0))
                                
                                # Skip unstable faces
                                if not is_stable or name == 'Scanning...':
                                    continue
                                
                                # Handle recognized faces
                                if name != "Unknown" and confidence >= Config.RECOGNITION_THRESHOLD:
                                    now = time.time()
                                    
                                    # Check if this is a new access attempt (past cooldown)
                                    is_new_access_attempt = (name not in self.last_access or 
                                                             (now - self.last_access[name]) > Config.COOLDOWN_SECONDS)
                                    
                                    if is_new_access_attempt:
                                        # For new access attempts, require blink verification
                                        if self.face_system.blink_detector and self.face_system.blink_detector.available and Config.ENABLE_BLINK_DETECTION:
                                            # Check if they have a valid (recent) blink verification
                                            blink_time = self.blink_verified.get(name, 0)
                                            blink_valid = (now - blink_time) < 3.0  # Blink must be within 3 seconds
                                            
                                            if blink_valid:
                                                # Blink was verified recently, grant access
                                                self.last_access[name] = now
                                                # Clear blink verification so next access requires new blink
                                                if name in self.blink_verified:
                                                    del self.blink_verified[name]
                                                self.root.after(0, lambda n=name, c=confidence: self.grant_access(n, c))
                                            # else: awaiting blink - handled by first pass that sets awaiting_blink_mode
                                        else:
                                            # Blink detection disabled, grant access directly
                                            self.last_access[name] = now
                                            self.root.after(0, lambda n=name, c=confidence: self.grant_access(n, c))
                                    # else: still in cooldown, don't grant again
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
                    # Skip face recognition entirely in admin mode to save compute
                    pass
                
                # Adaptive frame rate based on face detection
                if self.is_scanning and not self.registration_mode and not self.admin_mode:
                    if self.faces_detected > 0:
                        # Faces detected - use high FPS for responsiveness
                        self.target_fps = Config.ACTIVE_FPS
                        self.no_face_frames = 0
                    else:
                        # No faces - count frames and switch to idle mode
                        self.no_face_frames += 1
                        if self.no_face_frames >= self.idle_threshold_frames:
                            self.target_fps = Config.IDLE_FPS
                else:
                    # Registration or admin mode - use idle FPS
                    self.target_fps = Config.IDLE_FPS
                
                # Periodic cache cleanup to prevent memory buildup
                now = time.time()
                if now - self.last_cache_cleanup > self.cache_cleanup_interval:
                    expired_count = self.face_system.face_cache.cleanup_expired()
                    if expired_count and expired_count > 0:
                        logger.debug(f"Cache cleanup: removed {expired_count} expired entries")
                    # Also clean up old last_access entries to prevent memory leak
                    self._cleanup_old_access_entries(now)
                    self.last_cache_cleanup = now
                
                # Run garbage collection less frequently (separate from cache cleanup)
                if now - self.last_gc_run > Config.GC_INTERVAL_SECONDS:
                    gc.collect()
                    self.last_gc_run = now
                
                # Store current frame reference (avoid unnecessary copy when possible)
                self.current_frame = frame
                
                # Update display
                self.root.after(0, lambda f=display_frame: self.display_frame(f))
                
                # Adaptive frame rate based on detection state
                loop_time = time.time() - loop_start
                target_frame_time = 1.0 / self.target_fps
                sleep_time = max(0.001, target_frame_time - loop_time)
                time.sleep(sleep_time)
            
        except cv2.error as e:
            logger.error(f"OpenCV camera error: {e}", exc_info=True)
            self.root.after(0, lambda: self.show_toast("Camera error - restarting...", "error"))
            self._attempt_camera_recovery()
        except OSError as e:
            logger.error(f"Camera OS error: {e}", exc_info=True)
            self.root.after(0, lambda: self.show_toast("Camera hardware error", "error"))
            self._attempt_camera_recovery()
        except (tk.TclError, RuntimeError) as e:
            logger.error(f"UI error in camera loop: {e}", exc_info=True)
        finally:
            try:
                self.camera.stop()
            except Exception:
                pass
    
    def _attempt_camera_recovery(self) -> None:
        """Attempt to recover from camera errors by restarting the camera."""
        try:
            time.sleep(2)  # Wait before retry
            if self.is_running:
                logger.info("Attempting camera recovery...")
                self.camera.stop()
                time.sleep(1)
                self.camera.start()
                logger.info("Camera recovered successfully")
        except (cv2.error, OSError) as recovery_error:
            logger.error(f"Camera recovery failed: {recovery_error}", exc_info=True)
    
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
        """Build the user management tab with list, pagination, and delete functionality.
        
        Args:
            parent: Parent frame for the tab content.
        
        Displays a paginated list of registered users with their photo counts
        and provides buttons for revoking, restoring, and deleting users.
        """
        # Pagination state for users
        self.users_current_page = 0
        self.users_total_pages = 1
        self.users_total_count = 0
        self.users_all_items = []  # Cache of all user items for pagination
        
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
        self.manage_listbox.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 4))
        
        # Pagination controls
        pagination_frame = tk.Frame(card, bg=Config.COLOR_CARD)
        pagination_frame.pack(fill=tk.X, padx=8, pady=(0, 8))
        
        self.users_prev_btn = tk.Button(
            pagination_frame,
            text="◀ Previous",
            font=(Config.FONT_FAMILY, 9),
            fg=Config.COLOR_SCANNING,
            bg=Config.COLOR_CARD,
            activeforeground=Config.COLOR_SCANNING,
            activebackground=Config.COLOR_CARD_SECONDARY,
            bd=0,
            cursor="hand2",
            command=self.users_prev_page
        )
        self.users_prev_btn.pack(side=tk.LEFT)
        
        self.users_page_label = tk.Label(
            pagination_frame,
            text="Page 1 of 1",
            font=(Config.FONT_FAMILY, 9),
            fg=Config.COLOR_TEXT_SECONDARY,
            bg=Config.COLOR_CARD
        )
        self.users_page_label.pack(side=tk.LEFT, expand=True)
        
        self.users_next_btn = tk.Button(
            pagination_frame,
            text="Next ▶",
            font=(Config.FONT_FAMILY, 9),
            fg=Config.COLOR_SCANNING,
            bg=Config.COLOR_CARD,
            activeforeground=Config.COLOR_SCANNING,
            activebackground=Config.COLOR_CARD_SECONDARY,
            bd=0,
            cursor="hand2",
            command=self.users_next_page
        )
        self.users_next_btn.pack(side=tk.RIGHT)
        
        self.refresh_manage_list()
        
        # Action buttons row 1 - for active users
        btn_frame1 = tk.Frame(parent, bg=Config.COLOR_BG)
        btn_frame1.pack(fill=tk.X, padx=10, pady=(8, 2))
        
        tk.Button(
            btn_frame1,
            text="Revoke Access",
            font=(Config.FONT_FAMILY, 9),
            fg=Config.COLOR_SCANNING,
            bg=Config.COLOR_BG,
            activeforeground=Config.COLOR_SCANNING,
            bd=0,
            cursor="hand2",
            command=self.revoke_person_access
        ).pack(side=tk.LEFT)
        
        tk.Button(
            btn_frame1,
            text="Delete + Photos",
            font=(Config.FONT_FAMILY, 9),
            fg=Config.COLOR_DENIED,
            bg=Config.COLOR_BG,
            activeforeground=Config.COLOR_DENIED,
            bd=0,
            cursor="hand2",
            command=self.delete_person_and_photos
        ).pack(side=tk.RIGHT)
        
        # Action buttons row 2 - restore revoked users
        btn_frame2 = tk.Frame(parent, bg=Config.COLOR_BG)
        btn_frame2.pack(fill=tk.X, padx=10, pady=(2, 8))
        
        tk.Button(
            btn_frame2,
            text="Restore Access",
            font=(Config.FONT_FAMILY, 9),
            fg=Config.COLOR_GRANTED,
            bg=Config.COLOR_BG,
            activeforeground=Config.COLOR_GRANTED,
            bd=0,
            cursor="hand2",
            command=self.restore_person_access
        ).pack(side=tk.LEFT)
    
    def create_log_tab(self, parent):
        """Build the access log tab with filtering and pagination."""
        # Pagination state
        self.log_current_page = 0
        self.log_total_pages = 1
        self.log_total_count = 0
        self.log_date_from = None
        self.log_date_to = None
        self.log_name_var = tk.StringVar(value="All")
        
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
        self.admin_log_listbox.pack(fill=tk.BOTH, expand=True, padx=8, pady=(8, 4))
        
        # Pagination controls
        pagination_frame = tk.Frame(card, bg=Config.COLOR_CARD)
        pagination_frame.pack(fill=tk.X, padx=8, pady=(0, 8))
        
        self.log_prev_btn = tk.Button(
            pagination_frame,
            text="◀ Previous",
            font=(Config.FONT_FAMILY, 9),
            fg=Config.COLOR_SCANNING,
            bg=Config.COLOR_CARD,
            activeforeground=Config.COLOR_SCANNING,
            activebackground=Config.COLOR_CARD_SECONDARY,
            bd=0,
            cursor="hand2",
            command=self.log_prev_page
        )
        self.log_prev_btn.pack(side=tk.LEFT)
        
        self.log_page_label = tk.Label(
            pagination_frame,
            text="Page 1 of 1",
            font=(Config.FONT_FAMILY, 9),
            fg=Config.COLOR_TEXT_SECONDARY,
            bg=Config.COLOR_CARD
        )
        self.log_page_label.pack(side=tk.LEFT, expand=True)
        
        self.log_next_btn = tk.Button(
            pagination_frame,
            text="Next ▶",
            font=(Config.FONT_FAMILY, 9),
            fg=Config.COLOR_SCANNING,
            bg=Config.COLOR_CARD,
            activeforeground=Config.COLOR_SCANNING,
            activebackground=Config.COLOR_CARD_SECONDARY,
            bd=0,
            cursor="hand2",
            command=self.log_next_page
        )
        self.log_next_btn.pack(side=tk.RIGHT)
        
        # Load initial entries with pagination
        self.refresh_log_page()
    
    def set_log_date_range(self, days_back):
        """Apply a quick date filter to the access log display."""
        from datetime import timedelta
        today = datetime.now().date()
        
        if days_back == 0:
            self.log_date_from = today
        else:
            self.log_date_from = today - timedelta(days=days_back)
        
        self.log_date_to = today
        self.log_current_page = 0  # Reset to first page when filter changes
        self.refresh_log_page()
    
    def clear_log_filter(self):
        """Remove all filters and show all log entries."""
        self.log_name_var.set("All")
        self.log_date_from = None
        self.log_date_to = None
        self.log_current_page = 0
        self.refresh_log_page()
    
    def log_prev_page(self):
        """Navigate to previous page of log entries."""
        if self.log_current_page > 0:
            self.log_current_page -= 1
            self.refresh_log_page()
    
    def log_next_page(self):
        """Navigate to next page of log entries."""
        if self.log_current_page < self.log_total_pages - 1:
            self.log_current_page += 1
            self.refresh_log_page()
    
    def refresh_log_page(self):
        """Refresh the log display with current page and filters."""
        entries, total_count, total_pages = self.access_log.get_paginated(
            page=self.log_current_page,
            page_size=Config.LOG_PAGE_SIZE,
            date_from=self.log_date_from,
            date_to=self.log_date_to,
            name_filter=None if self.log_name_var.get() == "All" else self.log_name_var.get()
        )
        
        self.log_total_count = total_count
        self.log_total_pages = total_pages
        
        # Update pagination controls
        self.log_page_label.config(text=f"Page {self.log_current_page + 1} of {total_pages} ({total_count} entries)")
        
        # Enable/disable navigation buttons
        self.log_prev_btn.config(
            state=tk.NORMAL if self.log_current_page > 0 else tk.DISABLED,
            fg=Config.COLOR_SCANNING if self.log_current_page > 0 else Config.COLOR_TEXT_TERTIARY
        )
        self.log_next_btn.config(
            state=tk.NORMAL if self.log_current_page < total_pages - 1 else tk.DISABLED,
            fg=Config.COLOR_SCANNING if self.log_current_page < total_pages - 1 else Config.COLOR_TEXT_TERTIARY
        )
        
        self.populate_log_listbox(entries)
    
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
        """Refresh the users list showing both active and revoked users.
        
        Active users are shown normally, revoked users are shown with
        a [REVOKED] prefix. Builds full list then displays current page.
        """
        # Build complete list of all users
        self.users_all_items = []
        self.person_map = {}
        self.person_status = {}
        
        # Get active users from trained model
        trained_names = self.face_system.get_trained_persons()
        name_counts = {}
        for name in self.face_system.known_names:
            name_counts[name] = name_counts.get(name, 0) + 1
        
        for name in sorted(trained_names):
            count = name_counts.get(name, 0)
            self.users_all_items.append({
                'name': name,
                'status': 'active',
                'count': count,
                'display': f"  ✓  {name}   •   {count} encodings"
            })
        
        # Get revoked users
        revoked_users = self.face_system.get_disabled_persons()
        for name, count in sorted(revoked_users):
            self.users_all_items.append({
                'name': name,
                'status': 'revoked',
                'count': count,
                'display': f"  ✗  {name}   •   {count} encodings [REVOKED]"
            })
        
        # Calculate pagination
        self.users_total_count = len(self.users_all_items)
        self.users_total_pages = max(1, (self.users_total_count + Config.USERS_PAGE_SIZE - 1) // Config.USERS_PAGE_SIZE)
        
        # Ensure current page is valid
        if self.users_current_page >= self.users_total_pages:
            self.users_current_page = max(0, self.users_total_pages - 1)
        
        self.refresh_users_page()
    
    def users_prev_page(self):
        """Navigate to previous page of users."""
        if self.users_current_page > 0:
            self.users_current_page -= 1
            self.refresh_users_page()
    
    def users_next_page(self):
        """Navigate to next page of users."""
        if self.users_current_page < self.users_total_pages - 1:
            self.users_current_page += 1
            self.refresh_users_page()
    
    def refresh_users_page(self):
        """Display the current page of users."""
        self.manage_listbox.delete(0, tk.END)
        self.person_map = {}
        self.person_status = {}
        
        # Calculate slice for current page
        start_idx = self.users_current_page * Config.USERS_PAGE_SIZE
        end_idx = start_idx + Config.USERS_PAGE_SIZE
        page_items = self.users_all_items[start_idx:end_idx]
        
        # Populate listbox with current page
        for local_idx, item in enumerate(page_items):
            self.person_map[local_idx] = item['name']
            self.person_status[local_idx] = item['status']
            self.manage_listbox.insert(tk.END, item['display'])
        
        if not page_items:
            self.manage_listbox.insert(tk.END, "  No users registered")
        
        # Update pagination controls
        self.users_page_label.config(
            text=f"Page {self.users_current_page + 1} of {self.users_total_pages} ({self.users_total_count} users)"
        )
        
        # Enable/disable navigation buttons
        self.users_prev_btn.config(
            state=tk.NORMAL if self.users_current_page > 0 else tk.DISABLED,
            fg=Config.COLOR_SCANNING if self.users_current_page > 0 else Config.COLOR_TEXT_TERTIARY
        )
        self.users_next_btn.config(
            state=tk.NORMAL if self.users_current_page < self.users_total_pages - 1 else tk.DISABLED,
            fg=Config.COLOR_SCANNING if self.users_current_page < self.users_total_pages - 1 else Config.COLOR_TEXT_TERTIARY
        )
    
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
    
    def revoke_person_access(self):
        """Revoke access for the selected active user.
        
        Moves the person's encodings to the disabled list. They can be
        restored later without needing to retrain.
        """
        selection = self.manage_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a person")
            return
        
        selected_index = selection[0]
        name = self.person_map.get(selected_index)
        status = self.person_status.get(selected_index, 'active')
        
        if name is None:
            messagebox.showerror("Error", "Could not identify selected person. Please refresh and try again.")
            return
        
        if status == 'revoked':
            messagebox.showinfo("Already Revoked", f"'{name}' is already revoked. Use 'Restore Access' to re-enable.")
            return
        
        if messagebox.askyesno("Revoke Access", f"Revoke access for '{name}'?\n\nEncodings will be preserved for easy restoration."):
            success, message = self.face_system.revoke_person_access(name)
            if success:
                logger.info(f"Access revoked: {message}")
                self.refresh_manage_list()
                self.update_info_label()
                self.show_toast(f"'{name}' access revoked", "success")
            else:
                logger.warning(f"Revoke failed: {message}")
                self.show_toast(f"Failed: {message}", "error")
    
    def restore_person_access(self):
        """Restore access for a previously revoked user.
        
        Moves their encodings back from the disabled list to the active model.
        No retraining required.
        """
        selection = self.manage_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a revoked person")
            return
        
        selected_index = selection[0]
        name = self.person_map.get(selected_index)
        status = self.person_status.get(selected_index, 'active')
        
        if name is None:
            messagebox.showerror("Error", "Could not identify selected person. Please refresh and try again.")
            return
        
        if status == 'active':
            messagebox.showinfo("Already Active", f"'{name}' already has access.")
            return
        
        if messagebox.askyesno("Restore Access", f"Restore access for '{name}'?"):
            success, message = self.face_system.restore_person_access(name)
            if success:
                logger.info(f"Access restored: {message}")
                self.refresh_manage_list()
                self.update_info_label()
                self.show_toast(f"'{name}' access restored", "success")
            else:
                logger.warning(f"Restore failed: {message}")
                self.show_toast(f"Failed: {message}", "error")
    
    def delete_person_and_photos(self):
        """Permanently delete a person and all their photos.
        
        Removes the person's encodings (from active or disabled list) AND
        deletes their image folder from the dataset. This action cannot be undone.
        """
        selection = self.manage_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a person")
            return
        
        selected_index = selection[0]
        name = self.person_map.get(selected_index)
        status = self.person_status.get(selected_index, 'active')
        
        if name is None:
            messagebox.showerror("Error", "Could not identify selected person. Please refresh and try again.")
            return
        
        if messagebox.askyesno("Delete Permanently", f"Permanently delete '{name}' and all their photos?\n\nThis cannot be undone!"):
            import shutil
            
            # Delete image folder from dataset
            person_folder = os.path.join(self.face_system.dataset_path, name)
            if os.path.exists(person_folder):
                shutil.rmtree(person_folder)
                logger.info(f"Deleted photo folder: {person_folder}")
            
            # Remove encodings based on current status
            if status == 'revoked':
                # Remove from disabled encodings
                if name in self.face_system.disabled_encodings:
                    del self.face_system.disabled_encodings[name]
                    self.face_system.save_disabled_encodings()
                    logger.info(f"Removed {name} from disabled encodings")
            else:
                # Remove from active model
                success, message = self.face_system.remove_person_from_model(name)
                if success:
                    logger.info(f"User deleted: {message}")
                else:
                    logger.warning(f"Delete warning: {message}")
            
            self.refresh_manage_list()
            self.update_info_label()
            self.show_toast(f"'{name}' permanently deleted", "success")
    
    # ==================== ACCESS LOG MANAGEMENT ====================
    
    def clear_access_log(self):
        """Clear all access log entries with user confirmation."""
        entry_count = self.access_log.get_total_count()
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
        
        # Clean up blink detector
        if self.face_system.blink_detector:
            self.face_system.blink_detector.close()
        
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
