import cv2
import os
from ultralytics import YOLO
import mediapipe as mp
import xml.etree.ElementTree as ET
import string
import random

# Load YOLOv8 model
model_path = 'yolov8n.pt'  # Using a pre-trained YOLOv8 model
model = YOLO(model_path)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.5)

# Directory to save results
bbox_output_dir = 'C:/Users/alimo/OneDrive/Documents/FRT/YOLO MediaPipe Pose Detection/YOLOv8-Mediapipe-Pose-Detection/Output/Detected_Bbox'
Handsup_output_dir = 'C:/Users/alimo/OneDrive/Documents/FRT/YOLO MediaPipe Pose Detection/YOLOv8-Mediapipe-Pose-Detection/Output/Hands_Up'
xml_dir = 'C:/Users/alimo/OneDrive/Documents/FRT/YOLO MediaPipe Pose Detection/YOLOv8-Mediapipe-Pose-Detection/Output/XML_Files'

video_dir = 'C:/Users/alimo/OneDrive/Documents/FRT/YOLO MediaPipe Pose Detection/YOLOv8-Mediapipe-Pose-Detection/Videos_data'  # Update with video path

os.makedirs(Handsup_output_dir, exist_ok=True)
os.makedirs(xml_dir, exist_ok=True)
os.makedirs(bbox_output_dir, exist_ok=True)

# Confidence threshold
confidence_threshold = 0.5

def get_pose_landmarks(image):
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        keypoints = [landmark for landmark in results.pose_landmarks.landmark if landmark.visibility > confidence_threshold]
        if len(keypoints) > len(results.pose_landmarks.landmark) / 2:
            return results.pose_landmarks
    return None

def classify_pose(landmarks):
    if not landmarks:
        return "No pose detected"

    landmarks_list = [landmark for landmark in landmarks.landmark]

    left_wrist = landmarks_list[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks_list[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    left_eye = landmarks_list[mp_pose.PoseLandmark.LEFT_EYE.value]
    right_eye = landmarks_list[mp_pose.PoseLandmark.RIGHT_EYE.value]
    nose = landmarks_list[mp_pose.PoseLandmark.NOSE.value]

    # Check if either wrist is above the eyes or nose
    if (left_wrist.y < left_eye.y or left_wrist.y < nose.y) and (right_wrist.y < right_eye.y or right_wrist.y < nose.y):
        return "Hands Up"
    else:
        return "Other Pose"

def create_xml_for_bounding_box(bounding_box, file_name, save_directory):
    """
    Create an XML file for a given bounding box and file name, and save it to the specified directory.

    Parameters:
    bounding_box (tuple): A tuple containing (xmin, ymin, xmax, ymax).
    file_name (str): The name of the file (without extension).
    save_directory (str): The directory where the XML file will be saved.

    Returns:
    str: The name of the created XML file.
    """
    # Ensure the save directory exists
    os.makedirs(save_directory, exist_ok=True)

    # Define the root element
    annotation = ET.Element('annotation')

    # Define the folder element
    folder = ET.SubElement(annotation, 'folder')
    folder.text = 'images'

    # Define the filename element
    filename = ET.SubElement(annotation, 'filename')
    filename.text = f"{file_name}.jpg"

    # Define the path element
    path = ET.SubElement(annotation, 'path')
    path.text = os.path.join('Output\XML_Files', f"{file_name}.jpg")

    # Define the source element
    source = ET.SubElement(annotation, 'source')
    database = ET.SubElement(source, 'database')
    database.text = 'Unknown'

    # Define the size element
    size = ET.SubElement(annotation, 'size')
    width = ET.SubElement(size, 'width')
    width.text = '1280'  # Placeholder value
    height = ET.SubElement(size, 'height')
    height.text = '720'  # Placeholder value
    depth = ET.SubElement(size, 'depth')
    depth.text = '3'  # Assuming RGB images

    # Define the segmented element
    segmented = ET.SubElement(annotation, 'segmented')
    segmented.text = '0'

    # Define the object element
    obj = ET.SubElement(annotation, 'object')
    name = ET.SubElement(obj, 'name')
    name.text = 'shooting player'
    pose = ET.SubElement(obj, 'pose')
    pose.text = 'Unspecified'
    truncated = ET.SubElement(obj, 'truncated')
    truncated.text = '0'
    difficult = ET.SubElement(obj, 'difficult')
    difficult.text = '0'
    bndbox = ET.SubElement(obj, 'bndbox')
    xmin = ET.SubElement(bndbox, 'xmin')
    xmin.text = str(int(bounding_box[0]))
    ymin = ET.SubElement(bndbox, 'ymin')
    ymin.text = str(int(bounding_box[1]))
    xmax = ET.SubElement(bndbox, 'xmax')
    xmax.text = str(int(bounding_box[2]))
    ymax = ET.SubElement(bndbox, 'ymax')
    ymax.text = str(int(bounding_box[3]))

    # Convert the tree to a string and write it to a file
    tree = ET.ElementTree(annotation)
    xml_file_name = os.path.join(save_directory, f"{file_name}.xml")
    tree.write(xml_file_name)
    
def detect_person(model, frame):
    results = model.predict(frame)
    detected_persons = []

    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy()

        for box, score, class_id in zip(boxes, scores, class_ids):
            if class_id == 0 and score > confidence_threshold:  # Class 0 is 'person' in COCO dataset
                x1, y1, x2, y2 = map(int, box)
                height = y2 - y1
                width = x2 - x1
                center_x = (x1 + x2) // 2
                x1 = center_x - height // 2
                x2 = center_x + height // 2
                x1 = max(0, x1)
                x2 = min(frame.shape[1], x2)
                cropped_person = frame[y1:y2, x1:x2]
                landmarks = get_pose_landmarks(cropped_person)
                pose_label = classify_pose(landmarks)
                if pose_label == "Hands Up":
                    print('Hands UP')
                    detected_persons.append((box, score, class_id, pose_label))
    return detected_persons

def generate_random_name(length=8):
    """ Generate a random string of letters and digits.
    """
    chars = string.ascii_letters + string.digits
    return ''.join(random.choice(chars) for _ in range(length))

def video_proscessor():
    # Process each video
    frame_counter = 0
    count=0

    for video_file in os.listdir(video_dir):
        if video_file.endswith('.mp4'):
            video_path = os.path.join(video_dir, video_file)
            cap = cv2.VideoCapture(video_path)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                count+=1
                detected_persons = detect_person(model, frame)

                for box, score, class_id, pose_label in detected_persons:
                    
                    file_name = f'{generate_random_name()}.jpg'
                    file_path = os.path.join(Handsup_output_dir,file_name)
                    cv2.imwrite(file_path, frame)
                    
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    bbox_file_path = os.path.join(bbox_output_dir, file_name)
                    cv2.imwrite(bbox_file_path, image_rgb)

                    # create_pascal_voc_annotation(file_name, frame.shape, bounding_boxes, xml_dir)
                    create_xml_for_bounding_box(box, file_name.replace('.jpg', ''), xml_dir)
                    frame_counter += 1
                print(count)
            cap.release()

if __name__== '__main__':
    video_proscessor()