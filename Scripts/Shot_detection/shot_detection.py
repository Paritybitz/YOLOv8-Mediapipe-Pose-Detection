# Imports
import cv2
import os
from ultralytics import YOLO
import mediapipe as mp
import xml.etree.ElementTree as ET
import string
import random


### Initialize Models

# Load YOLOv8 model
model_path = 'yolov8n.pt'  # Using a pre-trained YOLOv8 model
model = YOLO(model_path)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.5)


### Directories

# Directory to save results
bbox_output_dir = 'C:/Users/alimo/OneDrive/Documents/FRT/YOLO MediaPipe Pose Detection/YOLOv8-Mediapipe-Pose-Detection/Output/Detected_Bbox'
Handsup_output_dir = 'C:/Users/alimo/OneDrive/Documents/FRT/YOLO MediaPipe Pose Detection/YOLOv8-Mediapipe-Pose-Detection/Output/Hands_Up'
xml_dir = 'C:/Users/alimo/OneDrive/Documents/FRT/YOLO MediaPipe Pose Detection/YOLOv8-Mediapipe-Pose-Detection/Output/XML_Files'

# (Video) input directory
video_dir = 'C:/Users/alimo/OneDrive/Documents/FRT/YOLO MediaPipe Pose Detection/YOLOv8-Mediapipe-Pose-Detection/Videos_data'  # Update with video path

os.makedirs(Handsup_output_dir, exist_ok=True)
os.makedirs(xml_dir, exist_ok=True)
os.makedirs(bbox_output_dir, exist_ok=True)


### Generate landmarks

confidence_threshold = 0.5

def get_pose_landmarks(image):
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        keypoints = [landmark for landmark in results.pose_landmarks.landmark if landmark.visibility > confidence_threshold]
        if len(keypoints) > len(results.pose_landmarks.landmark) / 2:
            return results.pose_landmarks
    return None


### Define essential landmarks

def classify_pose(landmarks):
    if not landmarks:
        return "No pose detected"
    
    # Landmark variable definitions

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


### Creating XML Files for each image

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