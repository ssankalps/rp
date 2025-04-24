import argparse
import warnings
from collections import OrderedDict
import os
import time
import numpy as np
import gc

import flwr as fl
import torch
import cv2
from torch.utils.data import Dataset, DataLoader
from datetime import datetime

# Setup for parsing command-line arguments
parser = argparse.ArgumentParser(description="Flower Federated Learning for Object Detection (Raspberry Pi)")

# Add arguments for server address and client ID
parser.add_argument(
    "--server_address",
    type=str,
    default="0.0.0.0:8080",
    help="gRPC server address (default '0.0.0.0:8080')",
)
parser.add_argument(
    "--cid",
    type=int,
    required=True,
    help="Client id. Should be an integer between 0 and NUM_CLIENTS",
)
parser.add_argument(
    "--rtsp_url",
    type=str,
    default="http://admin:JUITMU@192.168.0.116:554/H.264/ch01/main/av_stream",
    help="RTSP URL for the camera stream",
)
parser.add_argument(
    "--frames",
    type=int,
    default=5,  # Reduced from 10 to 5 for Raspberry Pi
    help="Number of frames to capture for evaluation",
)
parser.add_argument(
    "--train_frames",
    type=int,
    default=3,  # Number of frames to capture for training
    help="Number of frames to capture for training",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=1,  # Limited training epochs for Raspberry Pi
    help="Number of local training epochs",
)
parser.add_argument(
    "--resolution",
    type=str,
    default="320x240",
    help="Frame resolution (default: 320x240)",
)
parser.add_argument(
    "--skip_training",
    action="store_true",
    help="Skip local training to save resources",
)
parser.add_argument(
    "--debug",
    action="store_true",
    help="Enable debug logging",
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Define number of clients
NUM_CLIENTS = 5

# Custom dataset for RTSP frames
class RTSPFrameDataset(Dataset):
    """Dataset for RTSP frames captured in headless mode."""
    
    def __init__(self, frames, labels=None, transform=None):
        self.frames = frames
        self.labels = labels  # Can be None for evaluation
        self.transform = transform
        
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        
        # Ensure frame is a proper numpy array
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)
            
        # Convert to RGB if it's in BGR format
        if frame.shape[2] == 3 and not isinstance(frame, torch.Tensor):
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
        if self.transform:
            frame = self.transform(frame)
        
        # For training, we need both frames and their annotations
        if self.labels is not None:
            return frame, self.labels[idx]
        
        # For evaluation, just return the frame in the format YOLOv5 expects
        return frame


# Function to capture frames from RTSP stream in headless mode
def capture_frames(rtsp_url, num_frames, resolution=(320, 240), debug=False):
    """Capture frames from RTSP stream in fully headless mode."""
    frames = []
    
    # Open RTSP stream
    print(f"Connecting to RTSP stream: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        print("Error: Could not open RTSP stream.")
        # Try with different API backends
        for backend in [cv2.CAP_FFMPEG, cv2.CAP_GSTREAMER]:
            print(f"Trying with different backend...")
            cap = cv2.VideoCapture(rtsp_url, backend)
            if cap.isOpened():
                print(f"Successfully connected with alternative backend")
                break
        
        if not cap.isOpened():
            print("All connection attempts failed.")
            return frames
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    
    print(f"Successfully connected to stream.")
    print(f"AUTO MODE: Will capture {num_frames} frames")
    
    frames_captured = 0
    last_capture_time = time.time()
    retry_count = 0
    max_retries = 10
    
    while frames_captured < num_frames:
        ret, frame = cap.read()
        
        if not ret:
            retry_count += 1
            print(f"Error reading frame from stream. Retry {retry_count}/{max_retries}")
            if retry_count >= max_retries:
                print("Max retries reached. Abandoning capture.")
                break
            time.sleep(1)
            continue
        
        # Reset retry counter on successful frame
        retry_count = 0
        
        # Resize frame for performance
        frame = cv2.resize(frame, resolution)
        
        # Auto capture with 3-second delay
        if time.time() - last_capture_time >= 3:
            # Store RGB frame (YOLOv5 expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Debug: show frame shape and type
            if debug:
                print(f"Frame shape: {rgb_frame.shape}, dtype: {rgb_frame.dtype}")
                
            frames.append(rgb_frame)
            frames_captured += 1
            print(f"Auto-capturing frame {frames_captured}/{num_frames}")
            last_capture_time = time.time()
    
    # Release resources
    cap.release()
    print(f"Captured {len(frames)} frames")
    
    # Check if any frames were captured
    if len(frames) == 0:
        print("WARNING: No frames were captured!")
        
    return frames


# Function to generate pseudo-labels for training
def generate_pseudo_labels(model, frames, conf_threshold=0.5, debug=False):
    """Generate pseudo-labels from model predictions for self-training."""
    labels = []
    model.eval()
    
    print("Generating pseudo-labels for captured frames")
    
    with torch.no_grad():
        for i, frame in enumerate(frames):
            print(f"Processing frame {i+1}/{len(frames)}")
            
            # Convert frame to tensor format if needed
            if not isinstance(frame, torch.Tensor):
                # Create a batch dimension and ensure RGB format
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
            else:
                frame_tensor = frame.unsqueeze(0) if frame.dim() == 3 else frame
                
            # Debug
            if debug:
                print(f"Frame tensor shape: {frame_tensor.shape}")
                
            # Get predictions
            results = model(frame_tensor)
            
            # Extract detections
            if hasattr(results, 'xyxy'):
                # Process results from detector 
                detections = results.xyxy[0].cpu().numpy()  # x1, y1, x2, y2, confidence, class
            else:
                # Handle direct model output format (may need adjustment)
                pred = results[0] if isinstance(results, list) else results
                if isinstance(pred, torch.Tensor):
                    if len(pred.shape) >= 2 and pred.shape[1] >= 6:  # Ensure proper shape
                        detections = pred.cpu().numpy()
                    else:
                        print(f"Warning: Unexpected prediction shape: {pred.shape}")
                        detections = np.array([])
                else:
                    print(f"Warning: Unexpected prediction type: {type(pred)}")
                    detections = np.array([])
            
            # Filter by confidence
            if len(detections) > 0:
                high_conf_detections = detections[detections[:, 4] >= conf_threshold]
                if debug:
                    print(f"Found {len(high_conf_detections)} high confidence detections")
            else:
                high_conf_detections = np.array([])
                if debug:
                    print("No detections found")
            
            # Store as labels in YOLOv5 format
            labels.append(high_conf_detections)
            
    return labels


# Function to convert detections to YOLOv5 format
def convert_to_yolo_format(detections, img_shape):
    """Convert [x1,y1,x2,y2,conf,cls] detections to YOLO format [cls,x_center,y_center,w,h]"""
    height, width = img_shape[:2]
    yolo_labels = []
    
    for det in detections:
        if len(det) >= 6:  # Ensure proper detection format
            x1, y1, x2, y2, _, cls = det[:6]
            
            # Convert bbox to normalized xywh format
            x_center = ((x1 + x2) / 2) / width
            y_center = ((y1 + y2) / 2) / height
            w = (x2 - x1) / width
            h = (y2 - y1) / height
            
            # Create YOLO format label
            yolo_label = [int(cls), x_center, y_center, w, h]
            yolo_labels.append(yolo_label)
            
    return np.array(yolo_labels)


# Function to train the model locally
def train_model(model, frames, labels, device, epochs=1, debug=False):
    """Train YOLOv5 model on local data with simplified hyperparameters for Pi."""
    print(f"Starting local training for {epochs} epochs")
    
    # Validate inputs
    if not frames or len(frames) == 0:
        print("Error: No frames available for training")
        return model
        
    if not labels or len(labels) == 0:
        print("Error: No labels available for training")
        return model
        
    # Check if any frames have detections
    has_detections = False
    for label_set in labels:
        if len(label_set) > 0:
            has_detections = True
            break
            
    if not has_detections:
        print("Warning: No objects detected in any frame. Training may not be effective.")
    
    # Set model to training mode
    model.train()
    
    # Use YOLOv5's built-in training engine with reduced hyperparameters
    # Create temporary dataset.yaml file with our frame data
    dataset_path = "./temp_dataset"
    os.makedirs(dataset_path, exist_ok=True)
    
    # Prepare data structure
    frames_dir = os.path.join(dataset_path, "images")
    labels_dir = os.path.join(dataset_path, "labels")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Save frames and labels in YOLOv5 format
    valid_examples = 0
    
    for i, (frame, detections) in enumerate(zip(frames, labels)):
        # Save image
        img_path = os.path.join(frames_dir, f"frame_{i}.jpg")
        
        # Ensure frame is in BGR format for saving
        if isinstance(frame, np.ndarray):
            if frame.shape[2] == 3:  # RGB format
                frame_to_save = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            else:
                frame_to_save = frame
        elif isinstance(frame, torch.Tensor):
            # Convert tensor to numpy array
            if frame.dim() == 3:  # CHW format
                frame_np = frame.permute(1, 2, 0).cpu().numpy()
            else:  # Already in HWC format
                frame_np = frame.cpu().numpy()
                
            # Scale to 0-255 if normalized
            if frame_np.max() <= 1.0:
                frame_np = (frame_np * 255).astype(np.uint8)
                
            frame_to_save = cv2.cvtColor(frame_np, cv2.COLOR_RGB2BGR)
        else:
            print(f"Warning: Unexpected frame type {type(frame)}")
            continue
            
        # Save the frame
        try:
            cv2.imwrite(img_path, frame_to_save)
            if debug:
                print(f"Saved frame to {img_path}")
        except Exception as e:
            print(f"Error saving frame: {e}")
            continue
        
        # Convert detections to YOLOv5 format and save labels
        label_path = os.path.join(labels_dir, f"frame_{i}.txt")
        
        try:
            with open(label_path, 'w') as f:
                if len(detections) == 0:
                    # Create empty file if no detections
                    pass
                else:
                    # Get image dimensions
                    height, width = frame.shape[:2]
                    
                    for det in detections:
                        if len(det) >= 6:  # Ensure it has x1,y1,x2,y2,conf,cls
                            x1, y1, x2, y2, _, cls = det[:6]
                            
                            # Convert bbox to normalized xywh format
                            x_center = ((x1 + x2) / 2) / width
                            y_center = ((y1 + y2) / 2) / height
                            w = (x2 - x1) / width
                            h = (y2 - y1) / height
                            
                            # Write to file: class x_center y_center width height
                            f.write(f"{int(cls)} {x_center} {y_center} {w} {h}\n")
                
                valid_examples += 1  # Count this as a valid example
        except Exception as e:
            print(f"Error processing labels: {e}")
            continue
    
    if valid_examples == 0:
        print("Error: No valid examples for training!")
        return model
        
    print(f"Prepared {valid_examples} valid training examples")
    
    # Create dataset.yaml file
    dataset_yaml = os.path.join(dataset_path, "dataset.yaml")
    with open(dataset_yaml, 'w') as f:
        f.write(f"path: {os.path.abspath(dataset_path)}\n")
        f.write(f"train: images\n")
        f.write(f"val: images\n")
        f.write(f"nc: 80\n")  # COCO classes
        f.write("names: ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']\n")
    
    # Train with lightweight hyperparameters
    try:
        # Get path to YOLOv5 training script
        import sys
        sys.path.append('./yolov5')  # Add YOLOv5 to path if needed
        
        # Simplified training command for Raspberry Pi resources
        print("Starting YOLOv5 training...")
        
        # Save current model state as starting point
        initial_weights = os.path.join(dataset_path, "initial_weights.pt")
        torch.save(model.state_dict(), initial_weights)
        
        # Use subprocess to call YOLOv5 train.py with minimal settings
        import subprocess
        train_cmd = [
            "python3", 
            "./yolov5/train.py",
            "--img", str(320),  # Fixed size for PI
            "--batch", "1",  # Small batch size for Pi
            "--epochs", str(epochs),
            "--data", dataset_yaml,
            "--weights", initial_weights,
            "--cache",
            "--device", "cpu",  # Force CPU
            "--save-period", "1",  # Save every epoch
            "--patience", "0",    # No early stopping
            "--nosave",           # Don't save intermediate results
            "--freeze", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",  # Freeze most layers
        ]
        
        print(f"Running command: {' '.join(train_cmd)}")
        result = subprocess.run(train_cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"Training error: {result.stderr}")
            
            # Fallback training method if YOLOv5 training fails
            print("Falling back to simplified training method")
            perform_simplified_training(model, frames, labels, device, epochs, debug)
    
    except Exception as e:
        print(f"Error during YOLOv5 training: {e}")
        print("Falling back to simplified training method")
        perform_simplified_training(model, frames, labels, device, epochs, debug)
    
    # Clean up temporary files
    try:
        import shutil
        shutil.rmtree(dataset_path)
    except Exception as e:
        print(f"Warning: Could not clean up temporary training files: {e}")
    
    return model


def perform_simplified_training(model, frames, labels, device, epochs=1, debug=False):
    """Perform simplified training directly using YOLOv5's detector and basic optimization."""
    print("Using simplified direct training method")
    
    # Check if there are valid frames and labels
    if not frames or len(frames) == 0:
        print("Error: No frames available for training")
        return model
        
    # Basic optimization settings for Pi
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), 
                               lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    
    # Track losses
    avg_loss = 0
    
    # Prepare frames and targets
    processed_frames = []
    processed_targets = []
    
    for i, (frame, detections) in enumerate(zip(frames, labels)):
        if debug:
            print(f"Processing frame {i} for training (shape: {frame.shape})")
            
        # Skip frames with zero detections - this is important!
        if len(detections) == 0:
            if debug:
                print(f"Skipping frame {i} - no detections")
            continue
            
        # Convert frame to tensor if needed
        if not isinstance(frame, torch.Tensor):
            # Create a batch dimension and ensure RGB format
            frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float().div(255.0)
        else:
            frame_tensor = frame
            
        # Convert detections to YOLOv5 format
        yolo_format_labels = convert_to_yolo_format(detections, frame.shape)
        
        if len(yolo_format_labels) > 0:
            target_tensor = torch.from_numpy(yolo_format_labels).float()
            
            processed_frames.append(frame_tensor)
            processed_targets.append(target_tensor)
            
            if debug:
                print(f"Added frame with {len(yolo_format_labels)} targets")
    
    # Check if we have any valid examples
    if len(processed_frames) == 0:
        print("No valid training examples found! Skipping training.")
        return model
        
    print(f"Prepared {len(processed_frames)} frames with valid detections for training")
    
    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0
        
        # Set model to training mode
        model.train()
        
        for i, (frame_tensor, target_tensor) in enumerate(zip(processed_frames, processed_targets)):
            print(f"  Training on frame {i+1}/{len(processed_frames)}")
            
            # Forward pass with native loss computation
            optimizer.zero_grad()
            
            # Add batch dimension if needed
            x = frame_tensor.unsqueeze(0) if frame_tensor.dim() == 3 else frame_tensor
            x = x.to(device)
            
            # Debug
            if debug:
                print(f"  Input tensor shape: {x.shape}")
            
            # Forward pass
            try:
                pred = model(x)
                
                # Use YOLOv5's built-in loss if available
                loss = None
                if hasattr(model, 'compute_loss'):
                    try:
                        # Format target for YOLOv5 loss function
                        target_list = [target_tensor.to(device)]
                        loss_dict = model.compute_loss(pred, target_list)
                        loss = loss_dict[0]  # Use the total loss
                    except Exception as e:
                        print(f"  Error computing loss: {e}")
                        loss = None
                
                # Fallback to simple loss
                if loss is None:
                    # Very simplified loss - MSE on coordinates
                    try:
                        # Extract bounding boxes from predictions
                        pred_boxes = pred[0] if isinstance(pred, tuple) else pred
                        if isinstance(pred_boxes, torch.Tensor):
                            # Create simple target tensor (just use raw detection coordinates)
                            box_preds = pred_boxes[..., :4]  # Get box coordinates
                            # Calculate simple MSE on first few predictions
                            dummy_target = torch.zeros_like(box_preds[:5])
                            loss = torch.nn.functional.mse_loss(box_preds[:5], dummy_target)
                    except Exception as e:
                        print(f"  Error in fallback loss: {e}")
                        continue
            except Exception as e:
                print(f"  Error in forward pass: {e}")
                continue
            
            # Skip if loss is None
            if loss is None:
                print("  Warning: Could not compute loss. Skipping this frame.")
                continue
                
            # Backward pass and optimize
            try:
                loss.backward()
                optimizer.step()
                
                # Track loss
                epoch_loss += loss.item()
                print(f"  Frame {i+1} loss: {loss.item():.6f}")
            except Exception as e:
                print(f"  Error in backward pass: {e}")
                continue
            
            # Free memory
            del x, pred, loss
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Step LR scheduler
        scheduler.step()
        
        # Average loss for epoch
        avg_epoch_loss = epoch_loss / len(processed_frames) if processed_frames else 0
        print(f"  Epoch {epoch+1} average loss: {avg_epoch_loss:.6f}")
        avg_loss += avg_epoch_loss
    
    # Return average loss across all epochs
    if epochs > 0:
        avg_loss /= epochs
        print(f"Training completed. Average loss: {avg_loss:.6f}")
    
    return model


# Object detection evaluation function
def evaluate_object_detection(model, frames, target_classes, class_names, device, debug=False):
    """Evaluate object detection on captured frames."""
    
    # Handle empty frame list
    if not frames or len(frames) == 0:
        print("No frames available for evaluation!")
        return {
            "loss": 1.0,
            "accuracy": 0.0,
            "mean_detections": 0.0,
            "detailed_results": [],
            "summary": {"total_counts": {}, "avg_confidences": {}}
        }
    
    # Set model to evaluation mode
    model.eval()
    
    # Initialize metrics
    total_counts = {cls_id: 0 for cls_id in target_classes}
    total_confidences = {cls_id: [] for cls_id in target_classes}
    
    results_data = []
    
    # Process each frame
    for i, frame in enumerate(frames):
        # Get timestamp
        frame_time = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        print(f"Evaluating frame {i+1}/{len(frames)} at {frame_time}")
        
        # Ensure frame is in correct format
        if not isinstance(frame, torch.Tensor):
            # Create a batch dimension and ensure RGB format
            x = torch.from_numpy(frame).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
        else:
            # Add batch dimension if needed
            x = frame.unsqueeze(0) if frame.dim() == 3 else frame
        
        # Move to device
        x = x.to(device)
        
        if debug:
            print(f"Frame tensor shape: {x.shape}")
        
        # Perform detection
        inf_start = time.time()
        with torch.no_grad():
            try:
                results = model(x)
            except Exception as e:
                print(f"Error during inference: {e}")
                continue
        inf_time = time.time() - inf_start
        print(f"Inference completed in {inf_time:.2f} seconds")
        
        # Get detections - handle different formats
        try:
            if hasattr(results, 'xyxy'):
                # Results from ultralytics YOLOv5
                detections = results.xyxy[0].cpu().numpy()
            else:
                # Raw output format
                pred = results[0] if isinstance(results, list) else results
                if isinstance(pred, torch.Tensor):
                    detections = pred.cpu().numpy()
                else:
                    print(f"Unknown results format: {type(results)}")
                    detections = np.array([])
        except Exception as e:
            print(f"Error extracting detections: {e}")
            detections = np.array([])
        
        # Reset frame counts and confidences
        frame_counts = {cls_id: 0 for cls_id in target_classes}
        frame_confidences = {cls_id: [] for cls_id in target_classes}
        
        # Process detections for this frame
        for detection in detections:
            if len(detection) >= 6:  # Ensure correct format
                cls_id = int(detection[5])
                confidence = float(detection[4])
                
                if cls_id in target_classes:
                    frame_counts[cls_id] += 1
                    frame_confidences[cls_id].append(confidence)
                    
                    # Update summary statistics
                    total_counts[cls_id] += 1
                    total_confidences[cls_id].append(confidence)
        
        # Store frame data
        frame_data = {"frame": i, "timestamp": frame_time, "counts": {}}
        for cls_id in target_classes:
            cls_name = class_names[cls_id]
            count = frame_counts[cls_id]
            avg_conf = 0.0
            if count > 0:
                avg_conf = sum(frame_confidences[cls_id]) / len(frame_confidences[cls_id])
            frame_data["counts"][cls_name] = {"count": count, "confidence": avg_conf}
        
        results_data.append(frame_data)
        
        # Force garbage collection after each frame to minimize memory usage
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Calculate summary metrics
    summary = {"total_counts": {}, "avg_confidences": {}}
    for cls_id in target_classes:
        cls_name = class_names[cls_id]
        total = total_counts[cls_id]
        avg_per_frame = total / len(frames) if len(frames) > 0 else 0
        
        avg_confidence = 0.0
        if total > 0:
            avg_confidence = sum(total_confidences[cls_id]) / len(total_confidences[cls_id])
        
        summary["total_counts"][cls_name] = total
        summary["avg_confidences"][cls_name] = avg_confidence
    
    # Calculate overall metrics for model quality assessment
    total_detections = sum(total_counts.values())
    avg_confidence = 0.0
    all_confidences = []
    for confs in total_confidences.values():
        all_confidences.extend(confs)
    
    if all_confidences:
        avg_confidence = sum(all_confidences) / len(all_confidences)
    
    # For federated learning, we need a single metric to optimize
    # Using average confidence as our "accuracy" metric
    accuracy = avg_confidence
    
    # Also calculate mean detections per frame as another useful metric
    mean_detections = total_detections / len(frames) if len(frames) > 0 else 0
    
    return {
        "loss": 1.0 - accuracy,  # Lower is better for FL
        "accuracy": float(accuracy),
        "mean_detections": float(mean_detections),
        "detailed_results": results_data,
        "summary": summary
    }


# Define the Flower client class for Object Detection (continued)
class ObjectDetectionClient(fl.client.NumPyClient):
    """A Flower client for YOLOv5-based object detection fully headless for Raspberry Pi."""

    def __init__(self, rtsp_url, frames_to_capture, train_frames, cid, resolution=(320, 240), epochs=1, skip_training=False, debug=False):
        self.rtsp_url = rtsp_url
        self.frames_to_capture = frames_to_capture
        self.train_frames = train_frames
        self.cid = cid  # Client ID
        self.resolution = resolution
        self.epochs = epochs
        self.skip_training = skip_training
        self.debug = debug
        self.device = torch.device("cpu")  # Force CPU for Raspberry Pi
        
        # Create cache directory for model files
        os.makedirs('./.cache', exist_ok=True)
        
        # Load YOLOv5n model
        print(f"Loading YOLOv5n model on device: {self.device}")
        try:
            # Try to use locally cached model to avoid repeated downloads
            if os.path.exists('./.cache/yolov5n.pt'):
                print("Loading model from local cache")
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='./.cache/yolov5n.pt', force_reload=False)
            else:
                print("Downloading YOLOv5n model (this may take a while)...")
                self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
                # Save model for future use
                torch.save(self.model.state_dict(), './.cache/yolov5n.pt')
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to direct loading with force_reload=True
            print("Trying alternative loading method...")
            try:
                self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True, force_reload=True)
            except Exception as e2:
                print(f"Second attempt failed: {e2}")
                print("Attempting final fallback method...")
                # Final fallback - clone repo and load model
                if not os.path.exists('yolov5'):
                    import subprocess
                    subprocess.run(['git', 'clone', 'https://github.com/ultralytics/yolov5'])
                
                # Load directly from local files
                sys.path.append('./yolov5')
                from models.common import DetectMultiBackend
                from utils.torch_utils import select_device
                
                self.model = DetectMultiBackend('./.cache/yolov5n.pt', device=select_device('cpu'))
        
        self.model.to(self.device)
        
        # Set confidence threshold
        self.model.conf = 0.5
        
        # Define target classes (from previous code)
        self.coco_class_mapping = {
            'person': 0,     # human
            'cup': 41,       # closest to mug
            'bottle': 39,    # bottle
            'chair': 56,     # chair
            'toothbrush': 79, # toothbrush
            'book': 73,      # book
            'keyboard': 66,  # keyboard
            'mouse': 64,     # mouse
            'scissors': 76,  # closest to pen/pencil
            'remote': 65,    # closest to hairbrush
        }
        self.target_classes = list(self.coco_class_mapping.values())
        self.class_names = {idx: name for name, idx in self.coco_class_mapping.items()}
    
    def set_parameters(self, params):
        """Set model weights from a list of NumPy ndarrays."""
        try:
            params_dict = zip(self.model.state_dict().keys(), params)
            state_dict = OrderedDict(
                {
                    k: torch.tensor(v) if v.size > 0 else torch.tensor([0])
                    for k, v in params_dict
                }
            )
            self.model.load_state_dict(state_dict, strict=False)
            print("Model parameters updated successfully")
        except Exception as e:
            print(f"Error updating model parameters: {e}")
            if self.debug:
                # Print more details to diagnose the issue
                model_keys = list(self.model.state_dict().keys())
                print(f"Model has {len(model_keys)} parameters")
                print(f"First few keys: {model_keys[:5]}")
                print(f"Received {len(params)} parameters")
                if len(params) > 0:
                    print(f"First param shape: {params[0].shape}")

    def get_parameters(self, config):
        """Get model weights as a list of NumPy ndarrays."""
        try:
            return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        except Exception as e:
            print(f"Error getting parameters: {e}")
            # Return empty parameters as fallback
            return [np.array([]) for _ in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """Training function with local training on Raspberry Pi."""
        print("Client sampled for fit()")
        
        # Set model parameters
        try:
            self.set_parameters(parameters)
        except Exception as e:
            print(f"Error setting parameters: {e}")
        
        # Skip training if requested
        if self.skip_training:
            print("Local training skipped as requested")
            return self.get_parameters({}), 0, {}
        
        # Capture frames for training
        print(f"Capturing {self.train_frames} frames for training...")
        train_frames = capture_frames(
            self.rtsp_url, 
            self.train_frames,
            resolution=self.resolution,
            debug=self.debug
        )
        
        if not train_frames or len(train_frames) == 0:
            print("No frames captured. Local training skipped.")
            return self.get_parameters({}), 0, {"train_loss": 0.0}
        
        # Generate pseudo-labels for self-supervised learning
        print("Generating pseudo-labels for training")
        pseudo_labels = generate_pseudo_labels(
            self.model, 
            train_frames, 
            debug=self.debug
        )
        
        # Check if we have any labels
        valid_labels = False
        for labels in pseudo_labels:
            if len(labels) > 0:
                valid_labels = True
                break
                
        if not valid_labels:
            print("No objects detected in any frame. Training would be ineffective. Skipping.")
            return self.get_parameters({}), len(train_frames), {"training_skipped": True}
        
        # Train the model locally
        print(f"Training model for {self.epochs} epochs")
        start_time = time.time()
        
        try:
            # Train model with captured frames and pseudo-labels
            train_model(
                model=self.model,
                frames=train_frames,
                labels=pseudo_labels,
                device=self.device,
                epochs=self.epochs,
                debug=self.debug
            )
            
            # Calculate training duration
            training_time = time.time() - start_time
            print(f"Local training completed in {training_time:.2f} seconds")
            
            # Clean up to save memory
            train_frames = None
            pseudo_labels = None
            gc.collect()
            
            # Get updated parameters after training
            updated_params = self.get_parameters({})
            
            # Save trained model weights
            try:
                model_path = f"client_{self.cid}_trained.pt"
                torch.save(self.model.state_dict(), model_path)
                print(f"Trained model saved at {model_path}")
            except Exception as e:
                print(f"Could not save trained model due to: {e}")
            
            # Return updated parameters and metrics
            return updated_params, self.train_frames, {"training_time": training_time}
            
        except Exception as e:
            print(f"Error during local training: {e}")
            print("Returning original parameters")
            return self.get_parameters({}), 0, {"train_error": str(e)}

    def evaluate(self, parameters, config):
        """Evaluation function optimized for Raspberry Pi."""
        print("Client sampled for evaluate()")
        
        # Set model parameters
        try:
            self.set_parameters(parameters)
        except Exception as e:
            print(f"Error setting parameters for evaluation: {e}")
        
        # Capture frames for evaluation
        print(f"Capturing {self.frames_to_capture} frames for evaluation...")
        frames = capture_frames(
            self.rtsp_url, 
            self.frames_to_capture,
            resolution=self.resolution,
            debug=self.debug
        )
        
        if not frames or len(frames) == 0:
            print("No frames captured. Evaluation failed.")
            return 0.0, 0, {"accuracy": 0.0, "error": "No frames captured"}
        
        # Evaluate object detection on captured frames
        metrics = evaluate_object_detection(
            model=self.model,
            frames=frames,
            target_classes=self.target_classes,
            class_names=self.class_names,
            device=self.device,
            debug=self.debug
        )
        
        # Save metrics to file
        round_num = config.get('round', 0)
        filename = f"client_{self.cid}_round_{round_num}_metrics.txt"
        self._save_metrics(metrics, filename)
        
        # Save model weights after each round (if space allows)
        try:
            model_path = f"client_{self.cid}_round_{round_num}.pt"
            torch.save(self.model.state_dict(), model_path)
            print(f"Model saved at {model_path}")
        except Exception as e:
            print(f"Could not save model due to: {e}")
        
        # Force garbage collection
        frames = None
        gc.collect()
        
        # Return evaluation metrics
        return float(metrics["loss"]), self.frames_to_capture, {
            "accuracy": float(metrics["accuracy"]), 
            "mean_detections": float(metrics["mean_detections"])
        }
    
    def _save_metrics(self, metrics, filename):
        """Save metrics to a formatted text file."""
        with open(filename, 'w') as f:
            # Write header
            f.write(f"YOLOv5n Object Detection Evaluation - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Client ID: {self.cid}\n\n")
            
            # Write summary metrics
            f.write(f"Overall Metrics:\n")
            f.write(f"Average Confidence (accuracy): {metrics['accuracy']:.4f}\n")
            f.write(f"Mean Detections per Frame: {metrics['mean_detections']:.2f}\n\n")
            
            # Write per-class summary
            f.write("Per-Class Summary:\n")
            f.write("Class,Total Detections,Avg Confidence\n")
            summary = metrics["summary"]
            for cls_name in self.class_names.values():
                total = summary["total_counts"].get(cls_name, 0)
                conf = summary["avg_confidences"].get(cls_name, 0.0)
                f.write(f"{cls_name},{total},{conf:.4f}\n")
            
            # Write detailed per-frame results
            f.write("\nDetailed Frame Results:\n")
            f.write("Frame,Timestamp," + ",".join([f"{name}_count,{name}_conf" for name in self.class_names.values()]) + "\n")
            
            for frame_data in metrics["detailed_results"]:
                frame_num = frame_data["frame"]
                timestamp = frame_data["timestamp"]
                row = [str(frame_num), timestamp]
                
                for cls_name in self.class_names.values():
                    count = frame_data["counts"].get(cls_name, {}).get("count", 0)
                    conf = frame_data["counts"].get(cls_name, {}).get("confidence", 0.0)
                    row.extend([str(count), f"{conf:.4f}"])
                
                f.write(",".join(row) + "\n")
            
            print(f"Metrics saved to {filename}")

    def to_client(self):
        """Convert NumPyClient to Client."""
        return fl.client.Client(self)


def main():
    """Main function to start the Flower client."""
    args = parser.parse_args()
    
    # Parse resolution
    if "x" in args.resolution:
        width, height = map(int, args.resolution.split("x"))
        resolution = (width, height)
    else:
        resolution = (320, 240)  # Default
    
    print(f"Using resolution: {resolution[0]}x{resolution[1]}")
    print(f"Fully headless mode enabled")
    print(f"Local training: {'Disabled' if args.skip_training else f'Enabled with {args.epochs} epoch(s)'}")
    print(f"Debug mode: {'Enabled' if args.debug else 'Disabled'}")
    
    # Add import sys before starting client to ensure yolov5 is in path
    import sys
    sys.path.append('./yolov5')
    
    # Start Flower client
    fl.client.start_client(
        server_address=args.server_address,
        client=ObjectDetectionClient(
            rtsp_url=args.rtsp_url,
            frames_to_capture=args.frames,
            train_frames=args.train_frames,
            cid=args.cid,
            resolution=resolution,
            epochs=args.epochs,
            skip_training=args.skip_training,
            debug=args.debug
        ),
    )


if __name__ == "__main__":
    main()
