import cv2
import tensorflow as tf
import numpy as np
import os
import time
import mysql.connector
from datetime import datetime
from pyzbar.pyzbar import decode
from object_detection.utils import config_util, label_map_util, visualization_utils as viz_utils
from object_detection.builders import model_builder
import requests
from bs4 import BeautifulSoup
import re

def get_text_from_url(url):
    try:
        response = requests.get(url, timeout=5)  # Fetch the webpage
        soup = BeautifulSoup(response.text, 'html.parser')
        extracted_text = soup.get_text().strip()  # Extract text content
        return extracted_text
    except Exception as e:
        print(f"Error fetching data from URL: {e}")
        return "Unknown"

# Initialize camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera failed to open.")
    exit()

# Directories and paths
WORKSPACE_PATH = 'Tensorflow/workspace'
MODEL_PATH = WORKSPACE_PATH + '/models/my_ssd_mobnet'
ANNOTATION_PATH = WORKSPACE_PATH + '/annotations'
CONFIG_PATH = MODEL_PATH + '/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH

# Load pipeline config and model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-7')).expect_partial()

category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH + '/label_map.pbtxt')


def get_text_from_url(url):
    try:
        response = requests.get(url, timeout=5)  # Fetch the webpage
        soup = BeautifulSoup(response.text, 'html.parser')

        # Extract all text and clean unnecessary spaces
        extracted_text = soup.get_text().strip()

        # Use regex to remove extra spaces, newlines, and unwanted text
        cleaned_text = re.sub(r'\s+', ' ', extracted_text)  # Replace multiple spaces with one

        # Remove "Powered By QRStuff.com" or any footer text
        cleaned_text = re.sub(r'Powered By QRStuff\.com.*', '', cleaned_text, flags=re.IGNORECASE).strip()

        # Remove the "Text" prefix if present
        cleaned_text = re.sub(r'^Text\s+', '', cleaned_text)  # Removes "Text" if it's at the beginning

        return cleaned_text
    except Exception as e:
        print(f"Error fetching data from URL: {e}")
        return "Unknown"



def detect_fn(image):
    image, shapes = detection_model.preprocess(image)
    prediction_dict = detection_model.predict(image, shapes)
    detections = detection_model.postprocess(prediction_dict, shapes)
    return detections


def insert_message_to_mysql(status, ChickNumber):
    try:
        connection = mysql.connector.connect(host='127.0.0.1', user='root', password='', database='test')
        cursor = connection.cursor()
        query = "INSERT INTO chickstatus (ChickNumber, status) VALUES (%s, %s)"
        cursor.execute(query, (ChickNumber, status))
        connection.commit()
        print(f"Inserted: {status} | QR: {ChickNumber}")
    except mysql.connector.Error as err:
        print(f"MySQL Error: {err}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        continue

    image_np = np.array(frame)
    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
    detections['num_detections'] = num_detections
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    image_np_with_detections = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_detections,
        detections['detection_boxes'],
        detections['detection_classes'] + 1,
        detections['detection_scores'],
        category_index,
        use_normalized_coordinates=True,
        min_score_thresh=0.5
    )

    ChickNumber = "Unknown"
    for code in decode(image_np):
        qr_data = code.data.decode('utf-8')

        if qr_data.startswith("http"):  # If QR contains a URL, fetch the actual text
            ChickNumber = get_text_from_url(qr_data)
        else:
            ChickNumber = qr_data  # If it's plain text, use it directly

        print(f"QR Code Data: {ChickNumber}")
        cv2.putText(image_np_with_detections, ChickNumber, (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    for i in range(num_detections):
        detected_class = detections['detection_classes'][i] + 1
        detected_score = detections['detection_scores'][i]
        if detected_score >= 0.5:
            status = "Healthy" if detected_class == 1 else "Sick"
            insert_message_to_mysql(status, ChickNumber)  # Insert into database with QR code data

    cv2.imshow("Detection & QR Scan", image_np_with_detections)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
