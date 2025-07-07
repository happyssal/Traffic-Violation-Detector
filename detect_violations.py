import cv2
import numpy as np
from ultralytics import YOLO
import time
import math
import os

MODEL_PATH = 'runs/detect/traffic_violation_detector6/weights/best.pt'

CONF_THRESHOLD = 0.54
RED_LIGHT_CONF_THRESHOLD = 0.37

GLOBAL_MIN_CONF_THRESHOLD = min(CONF_THRESHOLD, RED_LIGHT_CONF_THRESHOLD)

IOU_THRESHOLD = 0.7

MIN_DETECTION_FRAMES = 3

CAR_CLASS_NAME = 'car'
RED_LIGHT_CLASS_NAME = 'traffic_light_red'

RED_LIGHT_PROXIMITY_THRESHOLD = 500
LINE_INTERSECTION_TOLERANCE = 5 

roi_points_list = []
selected_roi_line = None 
roi_selection_complete = False
temp_frame_for_roi = None

def mouse_callback(event, x, y, flags, param):
    global roi_points_list, roi_selection_complete, temp_frame_for_roi

    if event == cv2.EVENT_LBUTTONDOWN:
        if not roi_selection_complete and len(roi_points_list) < 2:
            roi_points_list.append((x, y))
            if temp_frame_for_roi is not None:
                cv2.circle(temp_frame_for_roi, (x, y), 5, (0, 255, 255), -1)
                if len(roi_points_list) == 2:
                    cv2.line(temp_frame_for_roi, roi_points_list[0], roi_points_list[1], (0, 255, 0), 2)
                cv2.imshow("Select Stop Line ROI", temp_frame_for_roi)

            if len(roi_points_list) == 2:
                roi_selection_complete = True
                print(f"Both points selected for stop line: {roi_points_list}")
                print("Press 'c' to confirm, 'r' to reset.")


try:
    model = YOLO(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    print(f"Error loading model from {MODEL_PATH}: {e}")
    print("Please ensure the model path is correct.")
    exit()

def intersect(p1, q1, p2, q2):
    
    def orientation(p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - \
              (q[0] - p[0]) * (r[1] - q[1])
        if val == 0: return 0 
        return 1 if val > 0 else 2 

    def onSegment(p, q, r):
        return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != 0 and o2 != 0 and o3 != 0 and o4 != 0 and o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and onSegment(p1, p2, q1): return True
    if o2 == 0 and onSegment(p1, q2, q1): return True
    if o3 == 0 and onSegment(p2, p1, q2): return True
    if o4 == 0 and onSegment(p2, q1, q2): return True

    return False 

def detect_and_check_violations(source_path, output_base_dir='output_videos'):
    global roi_points_list, selected_roi_line, roi_selection_complete, temp_frame_for_roi

    cap = cv2.VideoCapture(source_path)

    if not cap.isOpened():
        print(f"Error: Could not open video source {source_path}")
        print("Please check if the video file path is correct or if your webcam is accessible.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30.0

    video_dir, video_full_name = os.path.split(source_path)
    video_name, video_ext = os.path.splitext(video_full_name)
    
    output_processed_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), output_base_dir)

    output_video_name = f"{video_name}_processed{video_ext}"
    output_video_path = os.path.join(output_processed_dir, output_video_name)

    if not os.path.exists(output_processed_dir):
        os.makedirs(output_processed_dir)
        print(f"Created output directory: {output_processed_dir}")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    if not video_writer.isOpened():
        print(f"Error: Could not open video writer for {output_video_path}")
        print("Please check if the output path is valid and you have write permissions.")
        cap.release()
        return

    ret, frame_for_roi = cap.read()
    if not ret:
        print("Error: Could not read first frame for ROI selection.")
        cap.release()
        return

    temp_frame_for_roi = frame_for_roi.copy()
    cv2.namedWindow("Select Stop Line ROI")
    cv2.setMouseCallback("Select Stop Line ROI", mouse_callback)

    print("\n--- Interactive Stop Line ROI Selection ---")
    print("Click 2 points on the image to define the stop line.")
    print("Press 'c' to confirm your selection.")
    print("Press 'r' to reset the selection and start over.")
    print("Press 'q' to quit the application.")

    while True:
        display_frame = temp_frame_for_roi.copy()
        
        if len(roi_points_list) < 2:
            cv2.putText(display_frame, f"Click point {len(roi_points_list) + 1} of 2", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        else:
            cv2.putText(display_frame, "Both points selected. Press 'c' to confirm.", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow("Select Stop Line ROI", display_frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('c'):
            if roi_selection_complete:
                selected_roi_line = (roi_points_list[0], roi_points_list[1]) 
                print(f"Stop Line ROI confirmed: {roi_points_list}")
                break
            else:
                print(f"Please select both 2 points. Currently {len(roi_points_list)} points selected.")
        elif key == ord('r'):
            roi_points_list = []
            roi_selection_complete = False
            temp_frame_for_roi = frame_for_roi.copy()
            print("ROI selection reset. Please select 2 points again.")
        elif key == ord('q'):
            print("ROI selection cancelled. Exiting.")
            cap.release()
            cv2.destroyAllWindows()
            exit()
    
    cv2.destroyWindow("Select Stop Line ROI")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    tracked_objects = {}

    print("\nStarting traffic violation detection loop...")
    print(f"Output video will be saved to: {output_video_path}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or error reading frame.")
            break
        frame_count += 1

        results = model.track(frame, conf=GLOBAL_MIN_CONF_THRESHOLD, iou=IOU_THRESHOLD, persist=True, tracker="bytetrack.yaml")

        current_frame_detected_ids = set()
        red_lights_in_frame = []
        
        processed_detections_for_frame = [] 

        for r in results:
            if r.boxes.data.numel() == 0:
                continue

            for i in range(len(r.boxes.xyxy)):
                x1_float, y1_float, x2_float, y2_float = r.boxes.xyxy[i].tolist()
                conf = r.boxes.conf[i].item()
                cls_id = int(r.boxes.cls[i].item())
                label = model.names[cls_id]

                if label == RED_LIGHT_CLASS_NAME:
                    if conf < RED_LIGHT_CONF_THRESHOLD:
                        continue
                elif label == CAR_CLASS_NAME:
                    if conf < CONF_THRESHOLD:
                        continue
                else: 
                    continue

                obj_id = None
                if r.boxes.id is not None and i < len(r.boxes.id):
                    obj_id = int(r.boxes.id[i].item())
                
                if obj_id is None:
                    x1, y1, x2, y2 = map(int, [x1_float, y1_float, x2_float, y2_float])
                    color = (100, 100, 100)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label} (No ID)", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2, cv2.LINE_AA)
                    cv2.putText(frame, f"Conf: {conf:.2f}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
                    continue

                processed_detections_for_frame.append({
                    'x1': x1_float, 'y1': y1_float, 'x2': x2_float, 'y2': y2_float,
                    'id': obj_id, 'conf': conf, 'label': label, 'cls_id': cls_id
                })
        
        if selected_roi_line is not None:
            cv2.line(frame, selected_roi_line[0], selected_roi_line[1], (255, 0, 0), 2)
            cv2.putText(frame, "Stop Line ROI", (selected_roi_line[0][0], selected_roi_line[0][1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)


        for det_data in processed_detections_for_frame:
            obj_id = det_data['id']
            label = det_data['label']
            x1_float, y1_float, x2_float, y2_float = det_data['x1'], det_data['y1'], det_data['x2'], det_data['y2']
            conf = det_data['conf']

            current_frame_detected_ids.add(obj_id)

            if label == RED_LIGHT_CLASS_NAME:
                red_lights_in_frame.append([x1_float, y1_float, x2_float, y2_float])

            if obj_id not in tracked_objects:
                tracked_objects[obj_id] = {
                    'violations': {
                        'red_light_cross': False
                    },
                    'class_label': label,
                    'frames_detected': 0
                }
            
            tracked_objects[obj_id]['frames_detected'] += 1

            if label == CAR_CLASS_NAME and tracked_objects[obj_id]['frames_detected'] >= MIN_DETECTION_FRAMES:
                car_box = [x1_float, y1_float, x2_float, y2_float]

                is_near_red_light = False
                for rl_box_float in red_lights_in_frame:
                    rl_center_x = (rl_box_float[0] + rl_box_float[2]) / 2
                    rl_center_y = (rl_box_float[1] + rl_box_float[3]) / 2
                    car_center_x = (car_box[0] + car_box[2]) / 2
                    car_center_y = (car_box[1] + car_box[3]) / 2
                    
                    dist_to_red_light = math.hypot(car_center_x - rl_center_x, car_center_y - rl_center_y)
                    
                    if dist_to_red_light < RED_LIGHT_PROXIMITY_THRESHOLD:
                        is_near_red_light = True
                        break

                is_crossing_stop_line = False
                if selected_roi_line is not None:
                    car_bottom_left = (int(x1_float), int(y2_float))
                    car_bottom_right = (int(x2_float), int(y2_float))
                    
                    if intersect(car_bottom_left, car_bottom_right, selected_roi_line[0], selected_roi_line[1]):
                        is_crossing_stop_line = True
                
                if is_near_red_light and is_crossing_stop_line:
                    if not tracked_objects[obj_id]['violations']['red_light_cross']:
                        print(f"Frame {frame_count}: VIOLATION DETECTED (ID: {obj_id}, Class: {label}): Red Light Stop Line Cross!")
                        tracked_objects[obj_id]['violations']['red_light_cross'] = True

            color = (0, 255, 0)
            text_color = (0, 0, 0)

            if obj_id in tracked_objects and tracked_objects[obj_id]['violations']['red_light_cross']:
                color = (0, 0, 255)
                
            x1, y1, x2, y2 = map(int, [x1_float, y1_float, x2_float, y2_float])
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            display_text = f"{label} ID:{obj_id}"
            if obj_id in tracked_objects and tracked_objects[obj_id]['violations']['red_light_cross']:
                display_text += " (VIOLATION!)"

            cv2.putText(frame, display_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2, cv2.LINE_AA)
            cv2.putText(frame, f"Conf: {conf:.2f}", (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)
        
        objects_to_expire = []
        for obj_id in tracked_objects:
            if obj_id not in current_frame_detected_ids:
                objects_to_expire.append(obj_id)
        
        for obj_id in objects_to_expire:
            del tracked_objects[obj_id]

        video_writer.write(frame)

        cv2.imshow("Traffic Violation Detector", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()
    print(f"Detection finished. Output video saved to {output_video_path}")


if __name__ == "__main__":
    input_video_path = input("Please enter the full path to the input video (e.g., C:/Users/YourName/video.mp4): ")
    
    if not os.path.exists(input_video_path):
        print(f"Error: The provided path '{input_video_path}' does not exist.")
        print("Please ensure the path is correct and the file exists.")
    elif not os.path.isfile(input_video_path):
        print(f"Error: The provided path '{input_video_path}' is not a file.")
        print("Please ensure you provide a valid video file path.")
    else:
        detect_and_check_violations(input_video_path)