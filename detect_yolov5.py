import cv2
import torch
import numpy as np
import time
from datetime import datetime

# Configuration
RTSP_URL = "http://admin:JUITMU@192.168.0.116:554/H.264/ch01/main/av_stream"  # Your EZVIZ camera RTSP URL
FRAMES_TO_CAPTURE = 100  # Number of frames to capture and analyze
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for detections
OUTPUT_FILE = "accuracy_before.txt"  # Output file for metrics
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main():
    print(f"Starting object detection using YOLOv5n on device: {DEVICE}")
    
    # Load YOLOv5n model
    model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
    model.to(DEVICE)
    model.conf = CONFIDENCE_THRESHOLD
    
    # COCO class indices for the requested objects
    # YOLOv5 uses COCO dataset with 80 classes
    # Here are the indices for your requested objects:
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
    
    # Initialize detection counters for each class
    class_counts = {cls_id: 0 for cls_id in target_classes}
    
    # Open RTSP stream
    print(f"Connecting to RTSP stream: {RTSP_URL}")
    cap = cv2.VideoCapture(RTSP_URL)
    
    if not cap.isOpened():
        print("Error: Could not open RTSP stream.")
        return
    
    print(f"Successfully connected to stream. Capturing {FRAMES_TO_CAPTURE} frames...")
    
    # Prepare output file
    with open(OUTPUT_FILE, 'w') as f:
        f.write(f"YOLOv5n Object Detection Results - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Classes monitored: {list(class_names.values())}\n")
        f.write("Frame,Timestamp," + ",".join(class_names.values()) + "\n")
    
    # Process frames
    frames_processed = 0
    
    while frames_processed < FRAMES_TO_CAPTURE:
        ret, frame = cap.read()
        
        if not ret:
            print("Error reading frame from stream. Retrying...")
            time.sleep(1)
            continue
        
        # Perform detection
        frame_time = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        results = model(frame)
        
        # Get detections
        detections = results.xyxy[0].cpu().numpy()  # x1, y1, x2, y2, confidence, class
        
        # Reset frame counts
        frame_counts = {cls_id: 0 for cls_id in target_classes}
        
        # Count detections for target classes
        for detection in detections:
            cls_id = int(detection[5])
            confidence = detection[4]
            
            if cls_id in target_classes and confidence >= CONFIDENCE_THRESHOLD:
                frame_counts[cls_id] += 1
                class_counts[cls_id] += 1
        
        # Log results for this frame
        with open(OUTPUT_FILE, 'a') as f:
            frame_data = [str(frames_processed), frame_time]
            for cls_id in target_classes:
                frame_data.append(str(frame_counts[cls_id]))
            f.write(",".join(frame_data) + "\n")
        
        # Display progress
        frames_processed += 1
        if frames_processed % 10 == 0:
            print(f"Processed {frames_processed}/{FRAMES_TO_CAPTURE} frames")
    
    # Write summary statistics
    with open(OUTPUT_FILE, 'a') as f:
        f.write("\nSummary Statistics:\n")
        f.write("Class,Total Detections,Avg Per Frame\n")
        for cls_id in target_classes:
            avg = class_counts[cls_id] / FRAMES_TO_CAPTURE
            f.write(f"{class_names[cls_id]},{class_counts[cls_id]},{avg:.2f}\n")
    
    # Release resources
    cap.release()
    print(f"Detection complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
