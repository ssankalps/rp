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
    "--resolution",
    type=str,
    default="320x240",
    help="Frame resolution (default: 320x240)",
)
parser.add_argument(
    "--headless",
    action="store_true",
    help="Run in headless mode without displaying frames",
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)

# Define number of clients
NUM_CLIENTS = 5

# Custom dataset for RTSP frames
class RTSPFrameDataset(Dataset):
    """Dataset for RTSP frames captured in interactive mode."""
    
    def __init__(self, frames, transform=None):
        self.frames = frames
        self.transform = transform
        
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        frame = self.frames[idx]
        if self.transform:
            frame = self.transform(frame)
        return {"images": frame}


# Function to capture frames from RTSP stream
def capture_frames(rtsp_url, num_frames, interactive=True, resolution=(320, 240), headless=False):
    """Capture frames from RTSP stream, either interactively or automatically."""
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
    if interactive and not headless:
        print("INTERACTIVE MODE: Press 'c' to capture a frame, 'q' to quit")
    else:
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
        
        # Display the frame if not in headless mode
        if not headless:
            try:
                cv2.imshow('RTSP Stream', frame)
            except Exception as e:
                print(f"Warning: Could not display frame ({e}). Continuing in headless mode.")
                headless = True
        
        # Interactive or auto capture logic
        capture_frame = False
        
        if interactive and not headless:
            # Wait for key press - 'c' to capture, 'q' to quit
            try:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    capture_frame = True
                    print(f"Capturing frame {frames_captured+1}/{num_frames}")
                elif key == ord('q'):
                    break
            except:
                # Fallback to auto mode if display not available
                interactive = False
                print("Switching to auto mode due to display issues")
        else:
            # Auto capture with 3-second delay
            if time.time() - last_capture_time >= 3:
                capture_frame = True
                print(f"Auto-capturing frame {frames_captured+1}/{num_frames}")
                last_capture_time = time.time()
        
        if capture_frame:
            # Store RGB frame (YOLOv5 expects RGB)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(rgb_frame)
            frames_captured += 1
    
    # Release resources
    cap.release()
    if not headless:
        try:
            cv2.destroyAllWindows()
        except:
            pass
    print(f"Captured {len(frames)} frames")
    
    return frames


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
    """A Flower client for YOLOv5-based object detection optimized for Raspberry Pi."""

    def __init__(self, rtsp_url, frames_to_capture, cid, resolution=(320, 240), headless=False):
        self.rtsp_url = rtsp_url
        self.frames_to_capture = frames_to_capture
        self.cid = cid  # Client ID
        self.resolution = resolution
        self.headless = headless
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
        """Training function.
        
        Note: For Raspberry Pi, we skip any local training to save resources.
        """
        print("Client sampled for fit()")
        self.set_parameters(parameters)
        
        # No local training on resource-constrained Raspberry Pi
        print("No local training performed on Raspberry Pi")
        
        # Force garbage collection
        gc.collect()
        
        # Return unchanged parameters and dataset size
        return self.get_parameters({}), 0, {}

    def evaluate(self, parameters, config):
        """Evaluation function optimized for Raspberry Pi."""
        print("Client sampled for evaluate()")
        self.set_parameters(parameters)
        
        # Capture frames for evaluation
        print(f"Capturing {self.frames_to_capture} frames for evaluation...")
        frames = capture_frames(
            self.rtsp_url, 
            self.frames_to_capture, 
            interactive=(not self.headless),
            resolution=self.resolution,
            headless=self.headless
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
    print(f"Headless mode: {'Enabled' if args.headless else 'Disabled'}")
    
    # Start Flower client
    fl.client.start_client(
        server_address=args.server_address,
        client=ObjectDetectionClient(
            rtsp_url=args.rtsp_url,
            frames_to_capture=args.frames,
            cid=args.cid,
            resolution=resolution,
            headless=args.headless
        ).to_client(),
    )


if __name__ == "__main__":
    main()
