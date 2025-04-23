import cv2
import torch
import numpy as np
import time
from datetime import datetime
import os

# Configuration
RTSP_URL = "http://admin:JUITMU@192.168.0.116:554/H.264/ch01/main/av_stream"
FRAMES_TO_CAPTURE = 10  # Reduced to 10 frames
CONFIDENCE_THRESHOLD = 0.5
OUTPUT_FILE = "accuracy_before.txt"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
CAPTURE_DELAY = 3  # Seconds between frame captures

# Set to False to run in headless mode with timed captures
SAVE_DETECTION_IMAGES = True  # Set whether to save detection images to disk
DETECTION_IMAGES_DIR = "detection_frames"  # Directory to save detection images

def main():
    print(f"Starting headless object detection using YOLOv5n on device: {DEVICE}")
    
    # Create directory for detection images if needed
    if SAVE_DETECTION_IMAGES:
        os.makedirs(DETECTION_IMAGES_DIR, exist_ok=True)
        print(f"Detection images will be saved to: {DETECTION_IMAGES_DIR}")
    
    # Load YOLOv5n model
    print("Loading YOLOv5n model...")
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
    model.to(DEVICE)
    model.conf = CONFIDENCE_THRESHOLD
    
    # COCO class mapping for everyday objects
    coco_class_mapping = {
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
    
    # Get the class names and indices we want to track
    target_classes = list(coco_class_mapping.values())
    class_names = {idx: name for name, idx in coco_class_mapping.items()}
    
    print(f"Tracking these 10 classes: {list(class_names.values())}")
    
    # Open RTSP stream
    print(f"Connecting to RTSP stream: {RTSP_URL}")
    cap = cv2.VideoCapture(RTSP_URL)
    
    if not cap.isOpened():
        print("Error: Could not open RTSP stream.")
        return
    
    print(f"Successfully connected to stream.")
    print(f"Will capture {FRAMES_TO_CAPTURE} frames with {CAPTURE_DELAY} seconds delay between each")
    
    # Prepare output file with header
    with open(OUTPUT_FILE, 'w') as f:
        f.write(f"YOLOv5n Object Detection Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Classes monitored: {list(class_names.values())}\n")
        
        # Create header row with count and confidence columns for each class
        header = ["Frame", "Timestamp"]
        for cls_name in class_names.values():
            header.extend([f"{cls_name}_count", f"{cls_name}_conf"])
        f.write(",".join(header) + "\n")
    
    # Initialize summary statistics
    total_counts = {cls_id: 0 for cls_id in target_classes}
    total_confidences = {cls_id: [] for cls_id in target_classes}
    
    # Process frames
    frames_processed = 0
    last_capture_time = time.time()
    
    while frames_processed < FRAMES_TO_CAPTURE:
        ret, frame = cap.read()
        
        if not ret:
            print("Error reading frame from stream. Retrying...")
            time.sleep(1)
            continue
        
        # Check if it's time to capture a new frame
        if frames_processed == 0 or time.time() - last_capture_time >= CAPTURE_DELAY:
            print(f"Capturing frame {frames_processed+1}/{FRAMES_TO_CAPTURE}")
            
            # Perform detection on the captured frame
            frame_time = datetime.now().strftime('%H:%M:%S.%f')[:-3]
            results = model(frame)
            
            # Get detections
            detections = results.xyxy[0].cpu().numpy()  # x1, y1, x2, y2, confidence, class
            
            # Reset frame counts and confidences
            frame_counts = {cls_id: 0 for cls_id in target_classes}
            frame_confidences = {cls_id: [] for cls_id in target_classes}
            
            # Process detections for this frame
            detection_frame = frame.copy() if SAVE_DETECTION_IMAGES else None
            
            for detection in detections:
                cls_id = int(detection[5])
                confidence = float(detection[4])
                
                if cls_id in target_classes and confidence >= CONFIDENCE_THRESHOLD:
                    frame_counts[cls_id] += 1
                    frame_confidences[cls_id].append(confidence)
                    
                    # Update summary statistics
                    total_counts[cls_id] += 1
                    total_confidences[cls_id].append(confidence)
                    
                    # Draw bounding box on the detection frame if saving images
                    if SAVE_DETECTION_IMAGES:
                        x1, y1, x2, y2 = int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3])
                        label = f"{class_names[cls_id]}: {confidence:.2f}"
                        cv2.rectangle(detection_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(detection_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Save the frame with detections if enabled
            if SAVE_DETECTION_IMAGES:
                output_path = os.path.join(DETECTION_IMAGES_DIR, f"frame_{frames_processed:03d}.jpg")
                cv2.imwrite(output_path, detection_frame)
                print(f"Saved detection image to {output_path}")
            
            # Log results for this frame
            with open(OUTPUT_FILE, 'a') as f:
                frame_data = [str(frames_processed), frame_time]
                
                for cls_id in target_classes:
                    # Add count
                    frame_data.append(str(frame_counts[cls_id]))
                    
                    # Calculate and add average confidence
                    avg_conf = 0.0
                    if frame_counts[cls_id] > 0:
                        avg_conf = sum(frame_confidences[cls_id]) / len(frame_confidences[cls_id])
                    frame_data.append(f"{avg_conf:.4f}")
                
                f.write(",".join(frame_data) + "\n")
            
            # Update counters
            frames_processed += 1
            last_capture_time = time.time()
    
    # Write summary statistics
    with open(OUTPUT_FILE, 'a') as f:
        f.write("\nSummary Statistics:\n")
        f.write("Class,Total Detections,Avg Per Frame,Avg Confidence\n")
        
        for cls_id in target_classes:
            # Calculate averages
            avg_per_frame = total_counts[cls_id] / frames_processed
            
            avg_confidence = 0.0
            if total_counts[cls_id] > 0:
                avg_confidence = sum(total_confidences[cls_id]) / len(total_confidences[cls_id])
            
            f.write(f"{class_names[cls_id]},{total_counts[cls_id]},{avg_per_frame:.2f},{avg_confidence:.4f}\n")
    
    # Release resources
    cap.release()
    print(f"Detection complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
