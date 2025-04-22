import cv2
import torch
import numpy as np
import time
from datetime import datetime
import os

# Configuration
RTSP_URL = "http://admin:JUITMU@192.168.0.116:554/H.264/ch01/main/av_stream"
FRAMES_TO_CAPTURE = 10
CONFIDENCE_THRESHOLD = 0.5
OUTPUT_FILE = "accuracy_before.txt"
DEVICE = torch.device('cpu')  # Force CPU on Raspberry Pi
CAPTURE_DELAY = 3  # Seconds between frame captures
FRAME_WIDTH = 320  # Reduced frame size for processing
FRAME_HEIGHT = 240

# Set to True for interactive mode (press 'c' to capture) or False for timed captures
INTERACTIVE_MODE = True

def main():
    print(f"Starting lightweight object detection for Raspberry Pi on {DEVICE}")
    
    # Create cache directory for downloaded model
    os.makedirs('./.cache', exist_ok=True)
    
    # Load YOLOv5n model - use smallest variant
    try:
        # Try to use locally cached model to avoid repeated downloads
        if os.path.exists('./.cache/yolov5n.pt'):
            print("Loading model from local cache")
            model = torch.hub.load('ultralytics/yolov5', 'custom', path='./.cache/yolov5n.pt')
        else:
            print("Downloading YOLOv5n model (this may take a while)...")
            model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True)
            # Save model for future use
            torch.save(model.state_dict(), './.cache/yolov5n.pt')
    except Exception as e:
        print(f"Error loading model: {e}")
        # Fallback to direct loading with force_reload=True
        print("Trying alternative loading method...")
        model = torch.hub.load('ultralytics/yolov5', 'yolov5n', pretrained=True, force_reload=True)
    
    # Optimize model for inference
    model.to(DEVICE)
    model.eval()  # Set to evaluation mode
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
    
    # Set lower resolution to improve performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    
    print(f"Successfully connected to stream.")
    if INTERACTIVE_MODE:
        print("INTERACTIVE MODE: Press 'c' to capture a frame, 'q' to quit")
    else:
        print(f"TIMED MODE: Will capture {FRAMES_TO_CAPTURE} frames with {CAPTURE_DELAY} seconds delay between each")
    
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
        
        # Resize frame to improve performance
        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        
        # Display the frame (disable for headless operation)
        try:
            cv2.imshow('RTSP Stream', frame)
        except:
            print("Display not available. Running in headless mode.")
        
        # Interactive or timed capture logic
        capture_frame = False
        
        if INTERACTIVE_MODE:
            # Wait for key press - 'c' to capture, 'q' to quit
            try:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('c'):
                    capture_frame = True
                    print(f"Capturing frame {frames_processed+1}/{FRAMES_TO_CAPTURE}")
                elif key == ord('q'):
                    break
            except:
                # Fallback for headless operation
                time.sleep(3)
                capture_frame = True
                print(f"Auto-capturing frame {frames_processed+1}/{FRAMES_TO_CAPTURE} (headless mode)")
        else:
            # Display frame and wait for delay period
            try:
                cv2.waitKey(1)
            except:
                pass
                
            if frames_processed == 0 or time.time() - last_capture_time >= CAPTURE_DELAY:
                capture_frame = True
                print(f"Capturing frame {frames_processed+1}/{FRAMES_TO_CAPTURE}")
                last_capture_time = time.time()
        
        # If we're not capturing this frame, continue to the next iteration
        if not capture_frame:
            continue
            
        # Get current timestamp before inference
        frame_time = datetime.now().strftime('%H:%M:%S.%f')[:-3]
        print(f"Starting inference on frame at {frame_time}...")
        
        # Inference start time
        inf_start = time.time()
        
        # Convert to RGB for YOLOv5
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Perform detection - use batched input for efficiency
        with torch.no_grad():  # Disable gradient calculations
            results = model(rgb_frame)
        
        # Inference time
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
            
            if cls_id in target_classes and confidence >= CONFIDENCE_THRESHOLD:
                frame_counts[cls_id] += 1
                frame_confidences[cls_id].append(confidence)
                
                # Update summary statistics
                total_counts[cls_id] += 1
                total_confidences[cls_id].append(confidence)
                
                # Draw bounding box on the displayed frame
                x1, y1, x2, y2 = int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3])
                label = f"{class_names[cls_id]}: {confidence:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Show the frame with detections
        try:
            cv2.imshow('Detection Result', frame)
            cv2.waitKey(1000)  # Display the result for 1 second
        except:
            print("Display not available. Continuing in headless mode.")
        
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
        if not INTERACTIVE_MODE:
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
    try:
        cv2.destroyAllWindows()
    except:
        pass
    print(f"Detection complete. Results saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
