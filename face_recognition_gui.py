"""
Face Recognition System with Tkinter GUI
========================================
A complete face recognition system with:
- Face Registration (capture images)
- Model Training
- Real-time Face Recognition

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
from PIL import Image, ImageTk

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
        time.sleep(0.5)  # Allow camera to warm up
    
    def capture_frame(self):
        """Capture a single frame from the camera"""
        if not self.is_running:
            return None
        
        if self.use_picamera:
            frame = self.camera.capture_array()
            # Convert XRGB to BGR for OpenCV
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


class FaceRecognitionSystem:
    """Core face recognition logic"""
    
    def __init__(self, dataset_path="dataset", encodings_path="encodings.pickle"):
        self.dataset_path = dataset_path
        self.encodings_path = encodings_path
        self.known_encodings = []
        self.known_names = []
        self.cv_scaler = 4  # Scale factor for faster processing
        
        # Ensure dataset folder exists
        if not os.path.exists(self.dataset_path):
            os.makedirs(self.dataset_path)
        
        # Load existing encodings if available
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
            
            # Extract person name from folder structure
            name = image_path.split(os.path.sep)[-2]
            
            # Load and convert image
            image = cv2.imread(image_path)
            if image is None:
                continue
            
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Detect faces and compute encodings
            boxes = face_recognition.face_locations(rgb, model="hog")
            encodings = face_recognition.face_encodings(rgb, boxes)
            
            for encoding in encodings:
                known_encodings.append(encoding)
                known_names.append(name)
        
        if not known_encodings:
            return False, "No faces detected in any images"
        
        # Save encodings
        data = {"encodings": known_encodings, "names": known_names}
        with open(self.encodings_path, "wb") as f:
            f.write(pickle.dumps(data))
        
        # Update loaded encodings
        self.known_encodings = known_encodings
        self.known_names = known_names
        
        return True, f"Training complete! {len(known_encodings)} face encodings from {len(set(known_names))} persons"
    
    def recognize_faces(self, frame):
        """Detect and recognize faces in a frame"""
        if not self.known_encodings:
            return frame, []
        
        # Resize for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=1/self.cv_scaler, fy=1/self.cv_scaler)
        rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find faces
        face_locations = face_recognition.face_locations(rgb_small)
        face_encodings = face_recognition.face_encodings(rgb_small, face_locations)
        
        results = []
        
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Scale back up
            top *= self.cv_scaler
            right *= self.cv_scaler
            bottom *= self.cv_scaler
            left *= self.cv_scaler
            
            # Compare with known faces
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
            
            # Draw rectangle and name
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label background
            cv2.rectangle(frame, (left, top - 35), (right, top), color, cv2.FILLED)
            
            # Draw name and confidence
            label = f"{name} ({confidence:.1%})" if name != "Unknown" else name
            cv2.putText(frame, label, (left + 6, top - 10), 
                       cv2.FONT_HERSHEY_DUPLEX, 0.6, (255, 255, 255), 1)
        
        return frame, results


class FaceRecognitionGUI:
    """Main GUI Application"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition System")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Initialize components
        self.camera = CameraManager(use_picamera=USE_PICAMERA, resolution=(640, 480))
        self.face_system = FaceRecognitionSystem()
        
        # State variables
        self.is_camera_active = False
        self.current_mode = None  # 'register', 'recognize', or None
        self.current_person_name = None
        self.captured_count = 0
        self.camera_thread = None
        self.stop_camera_flag = False
        
        # FPS calculation
        self.frame_count = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Setup GUI
        self.setup_styles()
        self.create_widgets()
        self.update_person_lists()
        
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def setup_styles(self):
        """Configure ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure colors
        style.configure('TFrame', background='#f0f0f0')
        style.configure('Header.TLabel', font=('Helvetica', 16, 'bold'), background='#f0f0f0')
        style.configure('SubHeader.TLabel', font=('Helvetica', 12, 'bold'), background='#f0f0f0')
        style.configure('Status.TLabel', font=('Helvetica', 10), background='#f0f0f0')
        
        style.configure('Action.TButton', font=('Helvetica', 11), padding=10)
        style.configure('Start.TButton', font=('Helvetica', 12, 'bold'), padding=15)
        style.configure('Stop.TButton', font=('Helvetica', 12, 'bold'), padding=15)
    
    def create_widgets(self):
        """Create all GUI widgets"""
        # Main container
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Left panel - Controls
        left_panel = ttk.Frame(main_frame, width=350)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Right panel - Camera feed
        right_panel = ttk.Frame(main_frame)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # === LEFT PANEL CONTENTS ===
        
        # Title
        title_label = ttk.Label(left_panel, text="Face Recognition System", style='Header.TLabel')
        title_label.pack(pady=(0, 20))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(left_panel)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Tab 1: Registration
        register_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(register_frame, text="📷 Register")
        self.create_register_tab(register_frame)
        
        # Tab 2: Training
        training_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(training_frame, text="🧠 Train")
        self.create_training_tab(training_frame)
        
        # Tab 3: Recognition
        recognition_frame = ttk.Frame(self.notebook, padding=10)
        self.notebook.add(recognition_frame, text="👤 Recognize")
        self.create_recognition_tab(recognition_frame)
        
        # === RIGHT PANEL CONTENTS ===
        
        # Camera display
        camera_label = ttk.Label(right_panel, text="Camera Feed", style='SubHeader.TLabel')
        camera_label.pack(pady=(0, 10))
        
        # Video frame with border
        video_frame = ttk.Frame(right_panel, relief='solid', borderwidth=2)
        video_frame.pack(fill=tk.BOTH, expand=True)
        
        self.video_label = ttk.Label(video_frame)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        
        # Status bar
        status_frame = ttk.Frame(right_panel)
        status_frame.pack(fill=tk.X, pady=(10, 0))
        
        self.status_label = ttk.Label(status_frame, text="Ready", style='Status.TLabel')
        self.status_label.pack(side=tk.LEFT)
        
        self.fps_label = ttk.Label(status_frame, text="FPS: --", style='Status.TLabel')
        self.fps_label.pack(side=tk.RIGHT)
        
        # Show placeholder image
        self.show_placeholder()
    
    def create_register_tab(self, parent):
        """Create the registration tab content"""
        # Instructions
        ttk.Label(parent, text="Register New Face", style='SubHeader.TLabel').pack(pady=(0, 10))
        ttk.Label(parent, text="Enter a name and capture multiple photos\nfrom different angles for best results.", 
                 wraplength=300, justify=tk.CENTER).pack(pady=(0, 15))
        
        # Name entry
        name_frame = ttk.Frame(parent)
        name_frame.pack(fill=tk.X, pady=5)
        ttk.Label(name_frame, text="Person Name:").pack(side=tk.LEFT)
        self.name_entry = ttk.Entry(name_frame, width=20)
        self.name_entry.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 0))
        
        # Capture count
        self.capture_count_label = ttk.Label(parent, text="Photos captured: 0")
        self.capture_count_label.pack(pady=10)
        
        # Buttons
        button_frame = ttk.Frame(parent)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.start_register_btn = ttk.Button(button_frame, text="▶ Start Camera", 
                                             command=self.start_registration, style='Start.TButton')
        self.start_register_btn.pack(fill=tk.X, pady=5)
        
        self.capture_btn = ttk.Button(button_frame, text="📸 Capture Photo", 
                                      command=self.capture_photo, style='Action.TButton', state='disabled')
        self.capture_btn.pack(fill=tk.X, pady=5)
        
        self.stop_register_btn = ttk.Button(button_frame, text="⏹ Stop", 
                                            command=self.stop_camera, style='Stop.TButton', state='disabled')
        self.stop_register_btn.pack(fill=tk.X, pady=5)
        
        # Registered persons list
        ttk.Separator(parent, orient='horizontal').pack(fill=tk.X, pady=15)
        ttk.Label(parent, text="Registered Persons:", style='SubHeader.TLabel').pack()
        
        list_frame = ttk.Frame(parent)
        list_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        scrollbar = ttk.Scrollbar(list_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.registered_listbox = tk.Listbox(list_frame, yscrollcommand=scrollbar.set, height=8)
        self.registered_listbox.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.registered_listbox.yview)
        
        # Delete button
        self.delete_person_btn = ttk.Button(parent, text="🗑 Delete Selected", 
                                            command=self.delete_person)
        self.delete_person_btn.pack(fill=tk.X, pady=5)
    
    def create_training_tab(self, parent):
        """Create the training tab content"""
        ttk.Label(parent, text="Train Recognition Model", style='SubHeader.TLabel').pack(pady=(0, 10))
        ttk.Label(parent, text="Train the model using all captured\nfaces in the dataset folder.", 
                 wraplength=300, justify=tk.CENTER).pack(pady=(0, 15))
        
        # Training info
        info_frame = ttk.LabelFrame(parent, text="Dataset Info", padding=10)
        info_frame.pack(fill=tk.X, pady=10)
        
        self.dataset_info_label = ttk.Label(info_frame, text="Loading...")
        self.dataset_info_label.pack()
        
        # Progress
        self.progress_frame = ttk.LabelFrame(parent, text="Training Progress", padding=10)
        self.progress_frame.pack(fill=tk.X, pady=10)
        
        self.progress_bar = ttk.Progressbar(self.progress_frame, mode='determinate', length=280)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        self.progress_label = ttk.Label(self.progress_frame, text="Ready to train")
        self.progress_label.pack()
        
        # Train button
        self.train_btn = ttk.Button(parent, text="🧠 Start Training", 
                                    command=self.start_training, style='Start.TButton')
        self.train_btn.pack(fill=tk.X, pady=15)
        
        # Trained model info
        ttk.Separator(parent, orient='horizontal').pack(fill=tk.X, pady=10)
        ttk.Label(parent, text="Trained Model:", style='SubHeader.TLabel').pack()
        
        self.model_info_label = ttk.Label(parent, text="No model loaded")
        self.model_info_label.pack(pady=10)
        
        # Reload button
        self.reload_btn = ttk.Button(parent, text="🔄 Reload Model", 
                                     command=self.reload_model, style='Action.TButton')
        self.reload_btn.pack(fill=tk.X)
    
    def create_recognition_tab(self, parent):
        """Create the recognition tab content"""
        ttk.Label(parent, text="Face Recognition", style='SubHeader.TLabel').pack(pady=(0, 10))
        ttk.Label(parent, text="Start real-time face recognition\nusing the trained model.", 
                 wraplength=300, justify=tk.CENTER).pack(pady=(0, 15))
        
        # Recognition controls
        self.start_recognize_btn = ttk.Button(parent, text="▶ Start Recognition", 
                                              command=self.start_recognition, style='Start.TButton')
        self.start_recognize_btn.pack(fill=tk.X, pady=10)
        
        self.stop_recognize_btn = ttk.Button(parent, text="⏹ Stop Recognition", 
                                             command=self.stop_camera, style='Stop.TButton', state='disabled')
        self.stop_recognize_btn.pack(fill=tk.X, pady=5)
        
        # Recognition log
        ttk.Separator(parent, orient='horizontal').pack(fill=tk.X, pady=15)
        ttk.Label(parent, text="Recognition Log:", style='SubHeader.TLabel').pack()
        
        log_frame = ttk.Frame(parent)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        log_scrollbar = ttk.Scrollbar(log_frame)
        log_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.recognition_log = tk.Text(log_frame, height=12, width=35, 
                                       yscrollcommand=log_scrollbar.set, state='disabled')
        self.recognition_log.pack(fill=tk.BOTH, expand=True)
        log_scrollbar.config(command=self.recognition_log.yview)
        
        # Clear log button
        ttk.Button(parent, text="🗑 Clear Log", command=self.clear_recognition_log).pack(fill=tk.X, pady=5)
    
    def show_placeholder(self):
        """Show a placeholder when camera is not active"""
        # Create a placeholder image
        placeholder = np.zeros((480, 640, 3), dtype=np.uint8)
        placeholder[:] = (50, 50, 50)
        
        # Add text
        text = "Camera Feed"
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_size = cv2.getTextSize(text, font, 1.5, 2)[0]
        text_x = (640 - text_size[0]) // 2
        text_y = (480 + text_size[1]) // 2
        cv2.putText(placeholder, text, (text_x, text_y), font, 1.5, (150, 150, 150), 2)
        
        self.display_frame(placeholder)
    
    def display_frame(self, frame):
        """Display a frame on the video label"""
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize to fit display if needed
        label_width = self.video_label.winfo_width()
        label_height = self.video_label.winfo_height()
        
        if label_width > 1 and label_height > 1:
            # Maintain aspect ratio
            frame_h, frame_w = frame_rgb.shape[:2]
            scale = min(label_width / frame_w, label_height / frame_h)
            new_w = int(frame_w * scale)
            new_h = int(frame_h * scale)
            frame_rgb = cv2.resize(frame_rgb, (new_w, new_h))
        
        # Convert to PIL Image and then to PhotoImage
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        
        self.video_label.imgtk = imgtk
        self.video_label.configure(image=imgtk)
    
    def update_person_lists(self):
        """Update the list of registered persons"""
        # Update registered persons listbox
        self.registered_listbox.delete(0, tk.END)
        persons = self.face_system.get_registered_persons()
        for name, count in persons:
            self.registered_listbox.insert(tk.END, f"{name} ({count} photos)")
        
        # Update dataset info
        total_persons = len(persons)
        total_images = sum(count for _, count in persons)
        self.dataset_info_label.config(text=f"{total_persons} persons, {total_images} total images")
        
        # Update model info
        trained_persons = self.face_system.get_trained_persons()
        if trained_persons:
            self.model_info_label.config(text=f"Model loaded: {len(trained_persons)} persons\n" + 
                                        ", ".join(trained_persons[:5]) + 
                                        ("..." if len(trained_persons) > 5 else ""))
        else:
            self.model_info_label.config(text="No model loaded or empty")
    
    def update_fps(self):
        """Calculate and update FPS display"""
        self.frame_count += 1
        elapsed = time.time() - self.fps_start_time
        if elapsed > 1:
            self.current_fps = self.frame_count / elapsed
            self.frame_count = 0
            self.fps_start_time = time.time()
            self.fps_label.config(text=f"FPS: {self.current_fps:.1f}")
    
    def start_registration(self):
        """Start camera for face registration"""
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showwarning("Warning", "Please enter a person's name first.")
            return
        
        self.current_person_name = name
        self.captured_count = 0
        self.capture_count_label.config(text=f"Photos captured: {self.captured_count}")
        
        self.current_mode = 'register'
        self.start_camera()
        
        # Update button states
        self.start_register_btn.config(state='disabled')
        self.capture_btn.config(state='normal')
        self.stop_register_btn.config(state='normal')
        self.name_entry.config(state='disabled')
        
        self.status_label.config(text=f"Registration mode - Capturing for: {name}")
    
    def start_recognition(self):
        """Start real-time face recognition"""
        if not self.face_system.known_encodings:
            messagebox.showwarning("Warning", "No trained model found. Please train the model first.")
            return
        
        self.current_mode = 'recognize'
        self.start_camera()
        
        # Update button states
        self.start_recognize_btn.config(state='disabled')
        self.stop_recognize_btn.config(state='normal')
        
        self.status_label.config(text="Recognition mode - Identifying faces...")
    
    def start_camera(self):
        """Start the camera in a background thread"""
        if self.is_camera_active:
            return
        
        self.stop_camera_flag = False
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.camera_thread.start()
    
    def camera_loop(self):
        """Main camera loop running in a separate thread"""
        try:
            self.camera.start()
            self.is_camera_active = True
            
            last_recognition_results = []
            
            while not self.stop_camera_flag:
                frame = self.camera.capture_frame()
                if frame is None:
                    continue
                
                # Process frame based on mode
                if self.current_mode == 'recognize':
                    frame, results = self.face_system.recognize_faces(frame)
                    
                    # Log new recognitions
                    for result in results:
                        if result['name'] != "Unknown":
                            self.log_recognition(result['name'], result['confidence'])
                else:
                    # Just display the frame with face detection boxes for registration
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
                    
                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    
                    # Add mode indicator
                    cv2.putText(frame, "REGISTRATION MODE", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    cv2.putText(frame, f"Person: {self.current_person_name}", (10, 60), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"Captured: {self.captured_count}", (10, 90), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, "Press 'Capture Photo' button", (10, 120), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                
                # Store current frame for capture
                self.current_frame = frame.copy()
                
                # Update display (must be done in main thread)
                self.root.after(0, lambda f=frame: self.display_frame(f))
                self.root.after(0, self.update_fps)
                
                time.sleep(0.03)  # ~30 FPS limit
            
        except Exception as e:
            print(f"Camera error: {e}")
            self.root.after(0, lambda: messagebox.showerror("Camera Error", str(e)))
        finally:
            self.camera.stop()
            self.is_camera_active = False
            self.root.after(0, self.show_placeholder)
    
    def capture_photo(self):
        """Capture and save current frame"""
        if not self.is_camera_active or self.current_mode != 'register':
            return
        
        if hasattr(self, 'current_frame') and self.current_frame is not None:
            filepath = self.face_system.save_face_image(self.current_frame, self.current_person_name)
            self.captured_count += 1
            self.capture_count_label.config(text=f"Photos captured: {self.captured_count}")
            self.status_label.config(text=f"Saved: {os.path.basename(filepath)}")
            self.update_person_lists()
    
    def stop_camera(self):
        """Stop the camera"""
        self.stop_camera_flag = True
        
        if self.camera_thread:
            self.camera_thread.join(timeout=2)
        
        # Reset button states based on mode
        if self.current_mode == 'register':
            self.start_register_btn.config(state='normal')
            self.capture_btn.config(state='disabled')
            self.stop_register_btn.config(state='disabled')
            self.name_entry.config(state='normal')
        elif self.current_mode == 'recognize':
            self.start_recognize_btn.config(state='normal')
            self.stop_recognize_btn.config(state='disabled')
        
        self.current_mode = None
        self.status_label.config(text="Ready")
        self.fps_label.config(text="FPS: --")
    
    def delete_person(self):
        """Delete selected person from dataset"""
        selection = self.registered_listbox.curselection()
        if not selection:
            messagebox.showwarning("Warning", "Please select a person to delete.")
            return
        
        item = self.registered_listbox.get(selection[0])
        name = item.split(" (")[0]
        
        if messagebox.askyesno("Confirm Delete", f"Delete all photos for '{name}'?\nThis cannot be undone."):
            import shutil
            person_folder = os.path.join(self.face_system.dataset_path, name)
            if os.path.exists(person_folder):
                shutil.rmtree(person_folder)
                self.update_person_lists()
                self.status_label.config(text=f"Deleted: {name}")
    
    def start_training(self):
        """Start model training in a background thread"""
        self.train_btn.config(state='disabled')
        self.progress_bar['value'] = 0
        self.progress_label.config(text="Starting training...")
        
        def training_thread():
            def progress_callback(current, total, filepath):
                progress = (current / total) * 100
                filename = os.path.basename(filepath)
                self.root.after(0, lambda: self.progress_bar.configure(value=progress))
                self.root.after(0, lambda: self.progress_label.config(
                    text=f"Processing {current}/{total}: {filename[:30]}..."))
            
            success, message = self.face_system.train_model(progress_callback)
            
            self.root.after(0, lambda: self.training_complete(success, message))
        
        thread = threading.Thread(target=training_thread, daemon=True)
        thread.start()
    
    def training_complete(self, success, message):
        """Handle training completion"""
        self.train_btn.config(state='normal')
        self.progress_bar['value'] = 100 if success else 0
        self.progress_label.config(text=message)
        
        if success:
            messagebox.showinfo("Training Complete", message)
        else:
            messagebox.showerror("Training Failed", message)
        
        self.update_person_lists()
    
    def reload_model(self):
        """Reload the trained model"""
        success = self.face_system.load_encodings()
        if success:
            self.update_person_lists()
            messagebox.showinfo("Success", "Model reloaded successfully!")
        else:
            messagebox.showwarning("Warning", "No model file found or error loading.")
    
    def log_recognition(self, name, confidence):
        """Log a recognition event"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {name} ({confidence:.1%})\n"
        
        self.recognition_log.config(state='normal')
        self.recognition_log.insert(tk.END, log_entry)
        self.recognition_log.see(tk.END)
        self.recognition_log.config(state='disabled')
    
    def clear_recognition_log(self):
        """Clear the recognition log"""
        self.recognition_log.config(state='normal')
        self.recognition_log.delete(1.0, tk.END)
        self.recognition_log.config(state='disabled')
    
    def on_closing(self):
        """Handle window close event"""
        self.stop_camera_flag = True
        if self.camera_thread:
            self.camera_thread.join(timeout=2)
        self.root.destroy()


def main():
    """Main entry point"""
    root = tk.Tk()
    app = FaceRecognitionGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
