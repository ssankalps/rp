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
        if self.transform:
            frame = self.transform(frame)
        
        # For training, we need both frames and their annotations
        if self.labels is not None:
            return frame, self.labels[idx]
        
        # For evaluation, just return the frame
        return {"images": frame}


# Function to capture frames from RTSP stream in headless mode
def capture_frames(rtsp_url, num_frames, resolution=(320, 240)):
    """Capture frames from RTSP stream in fully headless mode."""
    frames = []
    
    # Open RTSP stream
    print(f"Connecting to RTSP stream: {rtsp_url}")
    cap = cv2.VideoCapture(rtsp_url)
    
    if not cap.isOpened():
        print("Error: Could not open RTSP stream.")
        return frames
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
    
    print(f"Successfully connected to stream.")
    print(f"AUTO MODE: Will capture {num_frames} frames")
    
    frames_captured = 0
    last_capture_time = time.time()
    
    while frames_captured < num_frames:
        ret, frame = cap.read()
        
        if not ret:
            print("Error reading frame from stream. Retrying...")
            time.sleep(1)
            continue
        
        # Resize frame for performance
        frame = cv2.resize(frame, resolution)
        
        # Auto capture with 3-second delay
        if time.time() - last_capture_time >= 3:
            # Store RGB frame (YOLOv5 expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
            frames_captured += 1
            print(f"Auto-capturing frame {frames_captured}/{num_frames}")
            last_capture_time = time.time()
    
    # Release resources
    cap.release()
    print(f"Captured {len(frames)} frames")
    
    return frames


# Function to generate pseudo-labels for training
def generate_pseudo_labels(model, frames, conf_threshold=0.5):
    """Generate pseudo-labels from model predictions for self-training."""
    labels = []
    model.eval()
    
    print("Generating pseudo-labels for captured frames")
    
    with torch.no_grad():
        for i, frame in enumerate(frames):
            print(f"Processing frame {i+1}/{len(frames)}")
            # Get predictions
            results = model(frame)
            
            # Extract detections
            detections = results.xyxy[0].cpu().numpy()  # x1, y1, x2, y2, confidence, class
            
            # Filter by confidence
            high_conf_detections = detections[detections[:, 4] >= conf_threshold]
            
            # Store as labels
            labels.append(high_conf_detections)
            
    return labels


# Function to train the model locally
def train_model(model, frames, labels, device, epochs=1):
    """Train YOLOv5 model on local data with simplified hyperparameters for Pi."""
    print(f"Starting local training for {epochs} epochs")
    
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
    for i, (frame, detections) in enumerate(zip(frames, labels)):
        # Save image
        img_path = os.path.join(frames_dir, f"frame_{i}.jpg")
        cv2.imwrite(img_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        
        # Convert detections to YOLOv5 format and save labels
        label_path = os.path.join(labels_dir, f"frame_{i}.txt")
        with open(label_path, 'w') as f:
            if len(detections) == 0:
                # Empty file if no detections
                pass
            else:
                height, width = frame.shape[:2]
                for det in detections:
                    # Convert bbox to normalized xywh format
                    x1, y1, x2, y2, _, cls = det[:6]
                    x_center = ((x1 + x2) / 2) / width
                    y_center = ((y1 + y2) / 2) / height
                    w = (x2 - x1) / width
                    h = (y2 - y1) / height
                    
                    # Write to file: class x_center y_center width height
                    f.write(f"{int(cls)} {x_center} {y_center} {w} {h}\n")
    
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
            "--img", str(model.yaml.get('img_size', 640)),
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
            perform_simplified_training(model, frames, labels, device, epochs)
    
    except Exception as e:
        print(f"Error during YOLOv5 training: {e}")
        print("Falling back to simplified training method")
        perform_simplified_training(model, frames, labels, device, epochs)
    
    # Clean up temporary files
    try:
        import shutil
        shutil.rmtree(dataset_path)
    except:
        print("Warning: Could not clean up temporary training files")
    
    return model


def perform_simplified_training(model, frames, labels, device, epochs=1):
    """Perform simplified training directly using YOLOv5's detector and basic optimization."""
    print("Using simplified direct training method")
    
    # Basic optimization settings for Pi
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    
    # Track losses
    avg_loss = 0
    
    # Training loop
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0
        
        for i, (frame, target) in enumerate(zip(frames, labels)):
            print(f"  Training on frame {i+1}/{len(frames)}")
            
            # Forward pass with native loss computation
            optimizer.zero_grad()
            
            # Process frame - convert to tensor if needed
            if not isinstance(frame, torch.Tensor):
                # Create a batch dimension and ensure RGB format
                x = torch.from_numpy(frame).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
                x = x.to(device)
            else:
                x = frame.to(device)
            
            # Forward pass
            pred = model(x)
            
            # Use YOLOv5's built-in loss if available
            loss = None
            try:
                # Try to access loss computation
                loss_dict = model.compute_loss(pred, target)
                loss = loss_dict[0]  # Use the total loss
            except:
                # Fallback - very simplified loss based on bounding box coordinates
                loss = torch.tensor(0.0, device=device)
                if len(target) > 0:
                    # Simple MSE loss between predictions and targets
                    # This is a very simplified approach
                    pred_boxes = pred[0]  # First output is usually bboxes
                    if isinstance(pred_boxes, torch.Tensor):
                        # Create tensor from numpy target
                        if not isinstance(target, torch.Tensor):
                            target_tensor = torch.from_numpy(target).float().to(device)
                        else:
                            target_tensor = target.to(device)
                        
                        # Simple MSE loss on raw outputs (not ideal but works as fallback)
                        pred_reshaped = pred_boxes.view(-1)[:target_tensor.numel()]
                        target_reshaped = target_tensor.view(-1)[:pred_reshaped.numel()]
                        loss = torch.nn.functional.mse_loss(pred_reshaped, target_reshaped)
            
            # Skip if loss is None
            if loss is None:
                print("  Warning: Could not compute loss. Skipping this frame.")
                continue
                
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Track loss
            epoch_loss += loss.item()
            
            # Free memory
            del x, pred, loss
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # Step LR scheduler
        scheduler.step()
        
        # Average loss for epoch
        avg_epoch_loss = epoch_loss / len(frames) if frames else 0
        print(f"  Epoch {epoch+1} average loss: {avg_epoch_loss:.6f}")
        avg_loss += avg_epoch_loss
    
    # Return average loss across all epochs
    if epochs > 0:
        avg_loss /= epochs
        print(f"Training completed. Average loss: {avg_loss:.6f}")
    
    return model


# Object detection evaluation function
def evaluate_object_detection(model, frames, target_classes, class_names, device):
    """Evaluate object detection on captured frames."""
    
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
        
        # Perform detection
        inf_start = time.time()
        with torch.no_grad():
            results = model(frame)
        inf_time = time.time() - inf_start
        print(f"Inference completed in {inf_time:.2f} seconds")
        
        # Get detections
        detections = results.xyxy[0].cpu().numpy()  # x1, y1, x2, y2, confidence, class
        
        # Reset frame counts and confidences
        frame_counts = {cls_id: 0 for cls_id in target_classes}
        frame_confidences = {cls_id: [] for cls_id in target_classes}
        
        # Process detections for this frame
        for detection in detections:
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


# Define the Flower client class for Object Detection
class ObjectDetectionClient(fl.client.NumPyClient):
    """A Flower client for YOLOv5-based object detection fully headless for Raspberry Pi."""

    def __init__(self, rtsp_url, frames_to_capture, train_frames, cid, resolution=(320, 240), epochs=1, skip_training=False):
        self.rtsp_url = rtsp_url
        self.frames_to_capture = frames_to_capture
        self.train_frames = train_frames
        self.cid = cid  # Client ID
        self.resolution = resolution
        self.epochs = epochs
        self.skip_training = skip_training
        self.device = torch.device("cpu")  # Force CPU for Raspberry Pi
        
        # Create cache directory for model files
        os.makedirs('./.cache', exist_ok=True)
        
        # Load YOLOv5n model
        print(f"Loading YOLOv5n model on device: {self.device}")
        try:
            # Try to use locally cached model to avoid repeated downloads
            if os.path.exists('./.cache/yolov5n.pt'):
                print("Loading model from local cache")
                self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='./.cache/yolov5n.pt')
            else:
                print("Downloading YOLOv5n model (this may take a while)...")
                self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
                # Save model for future use
                torch.save(self.model.state_dict(), './.cache/yolov5n.pt')
        except Exception as e:
            print(f"Error loading model: {e}")
            # Fallback to direct loading with force_reload=True
            print("Trying alternative loading method...")
            self.model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True, force_reload=True)
        
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
                    k: torch.Tensor(v) if v.shape != torch.Size([]) else torch.Tensor([0])
                    for k, v in params_dict
                }
            )
            self.model.load_state_dict(state_dict, strict=True)
            print("Model parameters updated successfully")
        except Exception as e:
            print(f"Error updating model parameters: {e}")

    def get_parameters(self, config):
        """Get model weights as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]

    def fit(self, parameters, config):
        """Training function with local training on Raspberry Pi."""
        print("Client sampled for fit()")
        self.set_parameters(parameters)
        
        # Skip training if requested
        if self.skip_training:
            print("Local training skipped as requested")
            return self.get_parameters({}), 0, {}
        
        # Capture frames for training
        print(f"Capturing {self.train_frames} frames for training...")
        train_frames = capture_frames(
            self.rtsp_url, 
            self.train_frames,
            resolution=self.resolution
        )
        
        if not train_frames:
            print("No frames captured. Local training skipped.")
            return self.get_parameters({}), 0, {"train_loss": 0.0}
        
        # Generate pseudo-labels for self-supervised learning
        print("Generating pseudo-labels for training")
        pseudo_labels = generate_pseudo_labels(self.model, train_frames)
        
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
                epochs=self.epochs
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
        self.set_parameters(parameters)
        
        # Capture frames for evaluation
        print(f"Capturing {self.frames_to_capture} frames for evaluation...")
        frames = capture_frames(
            self.rtsp_url, 
            self.frames_to_capture,
            resolution=self.resolution
        )
        
        if not frames:
            print("No frames captured. Evaluation failed.")
            return 0.0, 0, {"accuracy": 0.0}
        
        # Evaluate object detection on captured frames
        metrics = evaluate_object_detection(
            model=self.model,
            frames=frames,
            target_classes=self.target_classes,
            class_names=self.class_names,
            device=self.device
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
        return float(metrics["loss"]), len(frames) if frames else 0, {
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
            skip_training=args.skip_training
        ).to_client(),
    )


if __name__ == "__main__":
    main()
