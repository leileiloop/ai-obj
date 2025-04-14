from flask import Flask, render_template, Response, jsonify, request, session, flash, url_for, redirect
#FlaskForm--> it is required to receive input from the user
# Whether uploading a video file  to our object detection model
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField,StringField,DecimalRangeField,IntegerRangeField
from werkzeug.utils import secure_filename
from wtforms.validators import InputRequired,NumberRange
import datetime, time
import os, sys
import cv2
import numpy as np
import tensorflow as tf
from object_detection.utils import config_util
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder
from datetime import datetime, timedelta
import http.client
import serial
from threading import Thread
import time
from plyer import notification
from flask_socketio import SocketIO, emit
import requests
import mysql.connector
from mysql.connector import Error
import json
from flask_cors import CORS
import smtplib
import mysql.connector
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import re
from bs4 import BeautifulSoup
from pyzbar.pyzbar import decode
import threading

global capture
capture=0

#make shots directory to save pics
try:
    os.mkdir('static/shots')
except OSError as error:
    pass

WORKSPACE_PATH = 'Tensorflow/workspace'
SCRIPTS_PATH = 'Tensorflow/scripts'
APIMODEL_PATH = 'Tensorflow/models'
ANNOTATION_PATH = WORKSPACE_PATH+'/annotations'
IMAGE_PATH = WORKSPACE_PATH+'/images'
MODEL_PATH = WORKSPACE_PATH+'/models'
PRETRAINED_MODEL_PATH = WORKSPACE_PATH+'/pre-trained-models'
CONFIG_PATH = MODEL_PATH+'/my_ssd_mobnet/pipeline.config'
CHECKPOINT_PATH = MODEL_PATH+'/my_ssd_mobnet/'

# Load pipeline config and build a detection model
configs = config_util.get_configs_from_pipeline_file(CONFIG_PATH)
detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# Restore checkpoint
ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
ckpt.restore(os.path.join(CHECKPOINT_PATH, 'ckpt-7')).expect_partial()

category_index = label_map_util.create_category_index_from_labelmap(ANNOTATION_PATH+'/label_map.pbtxt')

app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'vince'
CORS(app, resources={r"/*": {"origins": "*"}})


#Use FlaskForm to get input video file  from user
class UploadFileForm(FlaskForm):
    #We store the uploaded video file path in the FileField in the variable file
    #We have added validators to make sure the user inputs the video in the valid format  and user does upload the
    #video when prompted to do so
    file = FileField("File",validators=[InputRequired()])
    submit = SubmitField("Run")


# -------------Camera-------------

# Path to the "shots" folder
shots_folder = os.path.join(os.path.dirname(__file__), 'static/shots')

# Get a list of image files in the "shots" folder
image_files = [f for f in os.listdir(shots_folder) if f.endswith(('jpg', 'jpeg', 'png', 'gif'))]

healthy_count = 0
sick_count = 0
Chick_Number = None  # Store scanned chick number globally
lock = threading.Lock()

def generate_frames_web():

    global healthy_count
    global sick_count
       

            

    @tf.function
    def detect_fn(image):
        image, shapes = detection_model.preprocess(image)
        prediction_dict = detection_model.predict(image, shapes)
        detections = detection_model.postprocess(prediction_dict, shapes)
        return detections

    # FPS Calculation Variables
    frame_count = 0
    start_time = time.time()

    cap = cv2.VideoCapture(0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    global capture

    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame. Retrying...")
                time.sleep(1)  # Wait briefly before retrying
                continue

            frame_count += 1  # Count frames

            image_np = np.array(frame)

            input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
            detections = detect_fn(input_tensor)

            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}
            detections['num_detections'] = num_detections
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            label_id_offset = 1
            image_np_with_detections = image_np.copy()

            # Visualize bounding boxes and labels on the image
            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes'] + label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.5,
                agnostic_mode=False)

            global Chick_Number

            ChickNumber = None
            for code in decode(image_np):
                qr_data = code.data.decode('utf-8')

                if qr_data.startswith("http"):  # If QR contains a URL, fetch the actual text
                    ChickNumber = get_text_from_url(qr_data)
                else:
                    ChickNumber = qr_data  # If it's plain text, use it directly

                with lock:  # Ensure thread-safe update
                    Chick_Number = ChickNumber

                print(f"QR Code Data: {ChickNumber}")


                cv2.putText(image_np_with_detections, ChickNumber, (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


                register_chick(ChickNumber)

            # Font settings for overlay
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (10, 25)
            font_scale = 1
            font_color = (255, 255, 255)
            font_thickness = 1
            
            # Calculate FPS
            end_time = time.time()
            fps = frame_count / (end_time - start_time)

            # Overlay FPS on the frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 1
            font_color = (0, 255, 0)
            font_thickness = 2
            org_fps = (10, 50)  # Position for FPS text

            cv2.putText(image_np_with_detections, f'FPS: {fps:.2f}', org_fps, font, font_scale, font_color, font_thickness, cv2.LINE_AA)

            for i in range(num_detections):
                detected_class = detections['detection_classes'][i] + label_id_offset
                detected_score = detections['detection_scores'][i]

                if detected_class == 1:  # Healthy egg
                    if detected_score >= 0.5:
                        healthy_count += 1
                        status = f"Healthy"
                        insert_message_to_mysql(status, ChickNumber)  # Insert into database with QR code data
                        current_datetime = datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')

                        rectangle_width = 800
                        rectangle_height = 600
                        cv2.rectangle(image_np_with_detections, org,
                                      (org[0] + rectangle_width, org[1] - rectangle_height),
                                      (0, 0, 0), -1)
                        cv2.putText(image_np_with_detections, f'Date and Time: {current_datetime}', org, font,
                                    font_scale, font_color, font_thickness, cv2.LINE_AA)
                        capture = 1

                elif detected_class == 2:  # Sick egg
                    if detected_score >= 0.5:
                        sick_count += 1
                        status = f"Sick"
                        insert_message_to_mysql(status, ChickNumber)  # Insert into database with QR code data
                        current_datetime = datetime.now().strftime('%Y-%m-%d %I:%M:%S %p')

                        rectangle_width = 800
                        rectangle_height = 600
                        cv2.rectangle(image_np_with_detections, org,
                                      (org[0] + rectangle_width, org[1] - rectangle_height),
                                      (0, 0, 0), -1)
                        cv2.putText(image_np_with_detections, f'Date and Time: {current_datetime}', org, font,
                                    font_scale, font_color, font_thickness, cv2.LINE_AA)
                        capture = 1

                if capture:
                    capture = 0
                    now = datetime.now()
                    filename = "shot_{}.png".format(str(now).replace(":", ''))
                    p = os.path.join('static', 'shots', filename)
                    cv2.imwrite(p, image_np_with_detections)

            check_harvest_time()  # Check if any chick is ready for harvest

            ret, buffer = cv2.imencode('.jpg', image_np_with_detections)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

        except (cv2.error, http.client.IncompleteRead) as e:
            print(f"An error occurred while processing the video stream: {e}")
            continue

# ---------------Login Form---------------


@app.route('/', methods=['GET', 'POST'])
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username and password:
            # Admin Login
            if username.lower() == 'admin' and password == 'admin':
                session['user_role'] = 'admin'
                session['email'] = "admin@yourdomain.com"  # Store admin email
                return redirect(url_for('dashboard'))

            # User Login (Check Database)
            connection = create_connection()
            if connection:
                try:
                    cursor = connection.cursor(dictionary=True)
                    query = "SELECT * FROM users WHERE username = %s AND password = %s"
                    cursor.execute(query, (username, password))
                    user = cursor.fetchone()

                    if user:
                        session['user_role'] = 'user'
                        session['email'] = user['Email']  # Store logged-in user email
                        send_email_notifications(user['Email'])  # Send email on login
                        return redirect(url_for('dashboard'))
                    else:
                        error = 'Invalid credentials'
                except Exception as e:
                    error = f"Error: {e}"
                finally:
                    cursor.close()
                    connection.close()
            else:
                error = "Database connection failed."
        else:
            error = "Please fill out all fields."

        return render_template('login.html', error=error)

    return render_template('login.html', error='')



@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        # Get form data
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')

        # Basic validation can be added here

        connection = create_connection()
        if connection:
            try:
                cursor = connection.cursor()
                # Insert into your users table (adjust field names as necessary)
                insert_query = "INSERT INTO users (Email, Username, Password) VALUES (%s, %s, %s)"
                cursor.execute(insert_query, (email, username, password))
                connection.commit()
                flash('Registration successful! Please login.')
                return redirect(url_for('login'))
            except Error as e:
                flash(f"An error occurred: {e}")
            finally:
                cursor.close()
                connection.close()
        else:
            flash("Failed to connect to the database.")
    return render_template('register.html')


@app.route('/admin_dashboard', methods=['GET', 'POST'])
def admin_dashboard():
    return render_template('admin-dashboard.html')

@app.route('/main_dashboard', methods=['GET', 'POST'])
def main_dashboard():
    return render_template('main-dashboard.html')

@app.route('/manage_users', methods=['GET', 'POST'])
def manage_users():
    return render_template('manage-users.html')

@app.route('/report', methods=['GET', 'POST'])
def report():
    return render_template('report.html')

# ---------------Dashboard---------------


@app.route("/dashboard", methods=['GET', 'POST'])
def dashboard():
    read_from_arduino()
    return render_template('dashboard.html', image_files=image_files, data=arduino_data)

# ---------------Growth Monitoring---------------


@app.route("/webcam", methods=['GET', 'POST'])
def webcam():
    read_from_arduino()
    return render_template('growth.html', image_files=image_files, data=arduino_data)


@app.route('/webapp')
def webapp():
    read_from_arduino()
    return Response(generate_frames_web(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/feeding", methods=['GET', 'POST'])
def feeding():
    read_from_arduino()
    return render_template('feed.html', image_files=image_files, data=arduino_data)


# ---------------Environment---------------


@app.route("/environment", methods=['GET', 'POST'])
def environment():
    read_from_arduino()
    return render_template('environment.html', image_files=image_files, data=arduino_data)
    

# ---------------Sanitization---------------


@app.route("/sanitization", methods=['GET', 'POST'])
def sanitization():
    read_from_arduino()
    return render_template('sanitization.html', image_files=image_files, data=arduino_data)

# ---------------Image List---------------


@app.route('/get_image_list')
def get_image_list():
    shots_folder = os.path.join(app.static_folder, 'shots')
    image_files = [f for f in os.listdir(shots_folder) if f.endswith(('.jpg', '.jpeg', '.png', '.gif'))]
    return jsonify(image_files)

# ---------------Capture Image---------------

@app.route('/requests1', methods=['POST'])
def capture():
    if request.method == 'POST' and 'click' in request.form and request.form['click'] == 'Capture':
        # Capture logic (replace this with your actual capture code)
        # For example, you can use OpenCV to capture a frame from the camera
        # and save it to the 'static/shots' directory

        # Sample code (make sure to adapt this to your actual capture logic)
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        filename = os.path.join(shots_folder, f'captured_image_{time.time()}.jpg')
        cv2.imwrite(filename, frame)
        cap.release()

        return 'Capture successful'
    else:
        return 'Invalid request'

@app.route('/requests',methods=['POST','GET'])
def tasks():
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture = 1

            print("Capture initiated")
    elif request.method == 'GET':
        return render_template('growth.html', image_files=image_files)
    return render_template('growth.html', image_files=image_files)

# Configure serial port to connect to Arduino
arduino_port = ''  # Update this to your Arduino's port
baud_rate = 9600
ser = serial.Serial()

# Data storage
arduino_data = {
    "Hum": 0,
    "Temp": 0,
    "Light1": "OFF",
    "Light2": "OFF",
    "Light3": "OFF",
    "Light4": "OFF",
    "ExhaustFan": "OFF",
    "Water": "OFF",
    "Food": "OFF",
    "Water_Weight": 0,
    "Food_Weight": 0,
    "Weight_1": 0,
    "Weight_2": 0,
    "Weight_3": 0,
    "Weight_4": 0,
    "Weight_5": 0,
    "Weight_6": 0,
    "Weight": 0,
    "Water_Level": 0,
    "Food_Level": 0,
    "UVLight": "OFF",
    "Conveyor": "OFF",
    "Sprinkle": "OFF"
}


@app.route("/stop_device", methods=["POST"])
def stop_device():
    try:
        data = request.json
        command = data.get("command")

        valid_commands = [
            "STOP_CONVEYOR", "STOP_UV_LIGHT", "STOP_SPRINKLE",
            "STOP_LIGHT1", "STOP_LIGHT2", "STOP_LIGHT3", "STOP_LIGHT4",
            "STOP_FOOD_SERVO", "STOP_WATER", "STOP_EXHAUST"
        ]

        if command in valid_commands:
            ser.write(f"{command}\n".encode())  # Send command to Arduino
            return jsonify({"message": f"{command.replace('_', ' ')} stopped successfully!"})
        else:
            return jsonify({"error": "Invalid command"})
    except Exception as e:
        return jsonify({"error": str(e)})

# Define weight threshold (adjust based on real-world data)
WEIGHT_THRESHOLD = -0.1  # Minimum weight to detect a chick (e.g., 100 grams)



# Function to read data from Arduino
def read_from_arduino():
    global arduino_data
    buffer = ""
    try:
        while ser.in_waiting > 0:
            line = ser.readline().decode('utf-8').strip()  # Read a line of data
            buffer += line  # Append to buffer
            if line.endswith("}"):  # Process only if a complete JSON object is detected
                try:
                    print(f"Processing: {buffer}")
                    new_data = json.loads(buffer)  # Parse JSON
                    arduino_data.update(new_data)  # Update the data dictionary
                    insert_data_to_mysql()
                    insert_data_to_mysql2()
                    insert_data_to_mysql3()
                    insert_data_to_mysql4()

                    # Check each weight sensor and insert if conditions are met
                    for i in range(1, 7):
                        insert_weight_data(f"Weight_{i}")

                except json.JSONDecodeError:
                    print(f"Invalid JSON: {buffer}")  # Handle invalid JSON
                buffer = ""  # Clear the buffer after processing
    except Exception as e:
        print(f"Error reading data: {e}")

# Flask route to serve data
@app.route('/data', methods=['GET'])
def get_data():
    read_from_arduino()  # Ensure the latest data is read
    response = jsonify(arduino_data)
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response


# Establish a connection to the MySQL database
def create_connection():
    try:
        connection = mysql.connector.connect(
            host='127.0.0.1',     # Replace with your host
            user='root',  # Replace with your MySQL username
            password='',  # Replace with your MySQL password
            database='test'  # Replace with your database name
        )
        if connection.is_connected():
            print("Connected to MySQL database")
            return connection
    except Error as e:
        print(f"Error: {e}")
        return None
        
# Environment

# Function to insert data to mysql
import time

last_insert_time = 0  # Global variable to track the last insert time
insert_interval = 5   # Interval in seconds between data insertions

def insert_data_to_mysql():
    global last_insert_time

    current_time = time.time()  # Get the current time in seconds since the epoch

    # Check if the required interval has passed
    if current_time - last_insert_time < insert_interval:
        return  # Exit the function without inserting data

    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor()
            # Prepare the SQL insert statement
            insert_query = """
                INSERT INTO sensordata (DateTime, Humidity, Temperature, Light1, Light2, Light3, Light4, ExhaustFan)
                VALUES (NOW(), %s, %s, %s, %s, %s, %s, %s)
            """
            # Extract sensor data from the arduino_data dictionary
            sensor_values = (
                arduino_data["Temp"],
                arduino_data["Hum"],
                "ON" if arduino_data["Lights"] == "ON" else "OFF",
                "ON" if arduino_data["Lights"] == "ON" else "OFF",
                "ON" if arduino_data["Lights"] == "ON" else "OFF",
                "ON" if arduino_data["Lights"] == "ON" else "OFF",
                "ON" if arduino_data["ExhaustFan"] == "ON" else "OFF"
            )
            # Log the data to be inserted for debugging
            print(f"Preparing to insert data: {sensor_values}")
            cursor.execute(insert_query, sensor_values)
            connection.commit()
            print("Data inserted successfully.")
            last_insert_time = current_time  # Update the last insert time
        except Error as e:
            print(f"MySQL Error: {e}")  # Log MySQL-specific errors
        except Exception as ex:
            print(f"General Error: {ex}")  # Log other errors
        finally:
            cursor.close()
            connection.close()
    else:
        print("Failed to establish database connection.")


# Sanitization

# Define previous states to track changes
previous_states = {
    "Conveyor": "OFF",
    "Sprinkle": "OFF",
    "UVLight": "OFF",
}

def insert_data_to_mysql2():
    global previous_states
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor()
            # Prepare the SQL insert statement
            insert_query = """
                INSERT INTO sensordata2 (DateTime, Conveyor, Sprinkle, UVLight)
                VALUES (NOW(), %s, %s, %s)
            """
            # Extract current sensor states from arduino_data
            current_states = {
                "Conveyor": arduino_data.get("Conveyor", "OFF"),
                "Sprinkle": arduino_data.get("Sprinkle", "OFF"),
                "UVLight": arduino_data.get("UVLight", "OFF"),
            }
            
            # Check if there are any changes in the state
            if current_states != previous_states:
                # Log the state change for debugging
                print(f"State changed, inserting: {current_states}")
                
                # Prepare the data tuple
                sensor_values = (
                    current_states["Conveyor"],
                    current_states["Sprinkle"],
                    current_states["UVLight"],
                )
                
                # Execute the query and commit
                cursor.execute(insert_query, sensor_values)
                connection.commit()
                print("Data inserted successfully(Sanitization).")
                
                # Update previous states
                previous_states = current_states.copy()
            else:
                print("No state change, skipping insert.")
        except Error as e:
            print(f"MySQL Error: {e}")
        except Exception as ex:
            print(f"General Error: {ex}")
        finally:
            cursor.close()
            connection.close()
    else:
        print("Failed to establish database connection.")



# Supplies Stock
# Define previous states to track changes for Food and Water
previous_states_food_water = {
    "Food": "OFF",
    "Water": "OFF",
}

def insert_data_to_mysql3():
    global previous_states_food_water
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor()
            # Prepare the SQL insert statement
            insert_query = """
                INSERT INTO sensordata1 (DateTime, Food, Water)
                VALUES (NOW(), %s, %s)
            """
            # Extract current sensor states from arduino_data
            current_states = {
                "Food": arduino_data.get("Food", "OFF"),
                "Water": arduino_data.get("Water", "OFF"),
            }
            
            # Check if there are any changes in the state
            if current_states != previous_states_food_water:
                # Log the state change for debugging
                print(f"State changed, inserting: {current_states}")
                
                # Prepare the data tuple
                sensor_values = (
                    current_states["Food"],
                    current_states["Water"],
                )
                
                # Execute the query and commit
                cursor.execute(insert_query, sensor_values)
                connection.commit()
                print("Data inserted successfully (Supplies Stock).")
                
                # Update previous states
                previous_states_food_water = current_states.copy()
            else:
                print("No state change, skipping insert.")
        except Error as e:
            print(f"MySQL Error: {e}")
        except Exception as ex:
            print(f"General Error: {ex}")
        finally:
            cursor.close()
            connection.close()
    else:
        print("Failed to establish database connection.")



def insert_data_to_mysql4():
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor()
            # Prepare the SQL insert statement
            insert_query = """
                INSERT INTO sensordata4 (Water_Level, Food_Level, DateTime)
                VALUES (%s, %s, NOW())
            """
            # Extract sensor data from the arduino_data dictionary
            sensor_values = (
                arduino_data["Water_Level"],
                arduino_data["Food_Level"]
            )
            # Log the data to be inserted for debugging
            print(f"Preparing to insert data4: {sensor_values}")
            cursor.execute(insert_query, sensor_values)
            connection.commit()
            print("Data inserted successfully.")
        except Error as e:
            print(f"MySQL Error: {e}")  # Log MySQL-specific errors
        except Exception as ex:
            print(f"General Error: {ex}")  # Log other errors
        finally:
            cursor.close()
            connection.close()
    else:
        print("Failed to establish database connection.")

# Environment
def get_data():
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            # Update query to order by timestamp (assumed column)
            select_query = """
            SELECT Temperature, Humidity, Light1, Light2, Light3, Light4, ExhaustFan
            FROM sensordata
            ORDER BY DateTime DESC
            LIMIT 1
            """
            cursor.execute(select_query)
            result = cursor.fetchone()
            print("Retrieved data:", result)
            return result
        except Error as e:
            print(f"Error: {e}")
        finally:
            cursor.close()
            connection.close()

# Retrieve all data from the database
def get_all_data():
    connection = create_connection()  # Ensure this function is implemented to establish a DB connection
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)  # Return rows as dictionaries
            # Query to fetch all rows ordered by DateTime
            select_query = """
                SELECT DateTime, Temperature, Humidity, Light1, Light2, Light3, Light4, ExhaustFan
                FROM sensordata
                ORDER BY DateTime DESC
                LIMIT 10
            """
            cursor.execute(select_query)
            results = cursor.fetchall()  # Fetch all rows
            for result in results:
                if result.get("DateTime"):
                    # Convert DateTime to 12-hour format with AM/PM
                    original_datetime = result["DateTime"]
                    formatted_datetime = datetime.strptime(str(original_datetime), "%Y-%m-%d %H:%M:%S").strftime(
                        "%Y-%m-%d %I:%M:%S %p")
                    result["DateTime"] = formatted_datetime
            print("Retrieved all data:", results)  # Debugging log
            return results
        except Error as e:
            print(f"Error: {e}")  # Log any errors
            return None
        finally:
            cursor.close()  # Close the cursor
            connection.close()  # Close the connection

@app.route('/get_all_data')
def fetch_all_data():
    data = get_all_data()
    if data:  # Check if data exists
        # Return all rows as JSON
        return jsonify(data)  # Just return the data, Flask will automatically format it as JSON
    else:
        return jsonify({'error': 'No data found'})  # Handle case with no data


# Define the route to fetch data
@app.route('/get_data')
def fetch_data():
    data = get_data()
    if data:
        # Return JSON with the necessary fields
        return jsonify({
            'Temp': data.get('Temperature', 'N/A'),
            'Hum': data.get('Humidity', 'N/A'),
            'Light1': data.get('Light1', 'N/A'),
            'Light2': data.get('Light2', 'N/A'),
            'Light3': data.get('Light3', 'N/A'),
            'Light4': data.get('Light4', 'N/A'),
            'ExhaustFan': data.get('ExhaustFan', 'N/A')
        })
    else:
        return jsonify({'error': 'No data found'})


# Growth Tracking
def get_data1():
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            select_query = """
            SELECT Weight
            FROM sensordata3
            ORDER BY DateTime DESC
            LIMIT 1
            """
            print(f"Executing query: {select_query}")  # Log the query
            cursor.execute(select_query)
            result = cursor.fetchone()
            print("Query Result:", result)  # Log the result fetched
            return result
        except Error as e:
            print(f"Error occurred in get_data1: {e}")
            return None
        finally:
            cursor.close()
            connection.close()
    else:
        print("Connection failed in get_data1.")
        return None



# Define the route to fetch data
@app.route('/get_data1')
def fetch_data1():
    data = get_data1()
    if data:
        # Return JSON with the necessary fields
        return jsonify({
            'Weight': data.get('Weight', 'N/A')

        })
    else:
        return jsonify({'error': 'No data found'})

# Retrieve all data from the database
def get_all_data1():
    connection = create_connection()  # Ensure this function is implemented to establish a DB connection
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)  # Return rows as dictionaries
            # Query to fetch all rows ordered by DateTime
            select_query = """
                SELECT DateTime, ChickNumber, Weight
                FROM sensordata3
                ORDER BY DateTime DESC
                LIMIT 10
            """
            cursor.execute(select_query)
            results = cursor.fetchall()  # Fetch all rows
            for result in results:
                if result.get("DateTime"):
                    # Convert DateTime to 12-hour format with AM/PM
                    original_datetime = result["DateTime"]
                    formatted_datetime = datetime.strptime(str(original_datetime), "%Y-%m-%d %H:%M:%S").strftime(
                        "%Y-%m-%d %I:%M:%S %p")
                    result["DateTime"] = formatted_datetime
            print("Retrieved all data:", results)  # Debugging log
            return results
        except Error as e:
            print(f"Error: {e}")  # Log any errors
            return None
        finally:
            cursor.close()  # Close the cursor
            connection.close()  # Close the connection

@app.route('/get_all_data1')
def fetch_all_data1():
    data = get_all_data1()
    if data:  # Check if data exists
        # Return all rows as JSON
        return jsonify(data)  # Just return the data, Flask will automatically format it as JSON
    else:
        return jsonify({'error': 'No data found'})  # Handle case with no data


# Sanitization

def get_data2():
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            select_query = """
            SELECT Conveyor, Sprinkle, UVLight
            FROM sensordata2
            ORDER BY DateTime DESC
            LIMIT 1
            """
            print(f"Executing query: {select_query}")  # Log the query being executed
            cursor.execute(select_query)
            result = cursor.fetchone()
            print(f"Query Result: {result}")  # Log the result fetched
            return result
        except Error as e:
            print(f"Error: {str(e)}")  # More detailed error message
        finally:
            cursor.close()
            connection.close()

@app.route('/get_data2')
def fetch_data2():
    data = get_data2()
    if data:
        # Return JSON with the necessary fields
        return jsonify({
            'Conveyor': data.get('Conveyor', 'N/A'),
            'Sprinkle': data.get('Sprinkle', 'N/A'),
            'UVLight': data.get('UVLight', 'N/A')
        })
    else:
        return jsonify({'error': 'No data found'})

# Retrieve all data from the database
def get_all_data2():
    connection = create_connection()  # Ensure this function is implemented to establish a DB connection
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)  # Return rows as dictionaries
            # Query to fetch all rows ordered by DateTime
            select_query = """
                SELECT DateTime, Conveyor, Sprinkle, UVLight
                FROM sensordata2
                ORDER BY DateTime DESC
                LIMIT 10
            """
            cursor.execute(select_query)
            results = cursor.fetchall()  # Fetch all rows
            for result in results:
                if result.get("DateTime"):
                    # Convert DateTime to 12-hour format with AM/PM
                    original_datetime = result["DateTime"]
                    formatted_datetime = datetime.strptime(str(original_datetime), "%Y-%m-%d %H:%M:%S").strftime(
                        "%Y-%m-%d %I:%M:%S %p")
                    result["DateTime"] = formatted_datetime
            print("Retrieved all data:", results)  # Debugging log
            return results
        except Error as e:
            print(f"Error: {e}")  # Log any errors
            return None
        finally:
            cursor.close()  # Close the cursor
            connection.close()  # Close the connection

@app.route('/get_all_data2')
def fetch_all_data2():
    data = get_all_data2()
    if data:  # Check if data exists
        # Return all rows as JSON
        return jsonify(data)  # Just return the data, Flask will automatically format it as JSON
    else:
        return jsonify({'error': 'No data found'})  # Handle case with no data

# Supplies Stock

# Retrieve all data from the database
def get_all_data3():
    connection = create_connection()  # Ensure this function is implemented to establish a DB connection
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)  # Return rows as dictionaries
            # Query to fetch all rows ordered by DateTime
            select_query = """
                SELECT DateTime, Food, Water
                FROM sensordata1
                ORDER BY DateTime DESC
                LIMIT 10
            """
            cursor.execute(select_query)
            results = cursor.fetchall()  # Fetch all rows
            for result in results:
                if result.get("DateTime"):
                    # Convert DateTime to 12-hour format with AM/PM
                    original_datetime = result["DateTime"]
                    formatted_datetime = datetime.strptime(str(original_datetime), "%Y-%m-%d %H:%M:%S").strftime(
                        "%Y-%m-%d %I:%M:%S %p")
                    result["DateTime"] = formatted_datetime
            print("Retrieved all data:", results)  # Debugging log
            return results
        except Error as e:
            print(f"Error: {e}")  # Log any errors
            return None
        finally:
            cursor.close()  # Close the cursor
            connection.close()  # Close the connection

@app.route('/get_all_data3')
def fetch_all_data3():
    data = get_all_data3()
    if data:  # Check if data exists
        # Return all rows as JSON
        return jsonify(data)  # Just return the data, Flask will automatically format it as JSON
    else:
        return jsonify({'error': 'No data found'})  # Handle case with no data


# Dashboard(Notifications)

# Retrieve all data from the database
def get_all_data4():
    connection = create_connection()  # Ensure this function is implemented to establish a DB connection
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)  # Return rows as dictionaries
            # Query to fetch all rows ordered by DateTime
            select_query = """
                SELECT DateTime, message
                FROM notifications
                ORDER BY DateTime DESC
                LIMIT 10
            """
            cursor.execute(select_query)
            results = cursor.fetchall()  # Fetch all rows
            for result in results:
                if result.get("DateTime"):
                    # Convert DateTime to 12-hour format with AM/PM
                    original_datetime = result["DateTime"]
                    formatted_datetime = datetime.strptime(str(original_datetime), "%Y-%m-%d %H:%M:%S").strftime(
                        "%Y-%m-%d %I:%M:%S %p")
                    result["DateTime"] = formatted_datetime
            print("Retrieved all data:", results)  # Debugging log
            return results
        except Error as e:
            print(f"Error: {e}")  # Log any errors
            return None
        finally:
            cursor.close()  # Close the cursor
            connection.close()  # Close the connection

@app.route('/get_all_data4')
def fetch_all_data4():
    data = get_all_data4()
    if data:  # Check if data exists
        # Return all rows as JSON
        return jsonify(data)  # Just return the data, Flask will automatically format it as JSON
    else:
        return jsonify({'error': 'No data found'})  # Handle case with no data

@app.route('/insert_notifications', methods=['POST'])
def insert_notifications():
    connection = None
    try:
        data = request.get_json()
        notifications = data.get('notifications', [])

        if not notifications:
            return jsonify({"success": False, "message": "No notifications to insert"})

        connection = create_connection()
        if connection:
            cursor = connection.cursor()
            for notification in notifications:
                insert_query = "INSERT INTO notifications (message) VALUES (%s)"
                cursor.execute(insert_query, (notification,))
            connection.commit()

            # **Send Email ONLY if a new notification is inserted**
            if session.get('email'):
                send_email_notifications(session.get('email'))

            return jsonify({"success": True, "message": "Notifications inserted successfully"})

        else:
            return jsonify({"success": False, "message": "Database connection failed"})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"success": False, "message": "Failed to insert notifications"})

    finally:
        if connection:
            cursor.close()
            connection.close()


# Retrieve all data from the database
def get_all_data5():
    connection = create_connection()  # Ensure this function is implemented to establish a DB connection
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)  # Return rows as dictionaries
            # Query to fetch all rows ordered by DateTime
            select_query = """
                SELECT DateTime, ChickNumber, status
                FROM chickstatus
                ORDER BY DateTime DESC
                LIMIT 10
            """
            cursor.execute(select_query)
            results = cursor.fetchall()  # Fetch all rows
            for result in results:
                if result.get("DateTime"):
                    # Convert DateTime to 12-hour format with AM/PM
                    original_datetime = result["DateTime"]
                    formatted_datetime = datetime.strptime(str(original_datetime), "%Y-%m-%d %H:%M:%S").strftime(
                        "%Y-%m-%d %I:%M:%S %p")
                    result["DateTime"] = formatted_datetime
            print("Retrieved all data:", results)  # Debugging log
            return results
        except Error as e:
            print(f"Error: {e}")  # Log any errors
            return None
        finally:
            cursor.close()  # Close the cursor
            connection.close()  # Close the connection


@app.route('/get_all_data5')
def fetch_all_data5():
    data = get_all_data5()
    if data:  # Check if data exists
        # Return all rows as JSON
        return jsonify(data)  # Just return the data, Flask will automatically format it as JSON
    else:
        return jsonify({'error': 'No data found'})  # Handle case with no data

# Supplies Stock


def get_data3():
    connection = create_connection()  # Ensure this function is implemented to establish a DB connection
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)  # Return rows as dictionaries
            # Query to fetch all rows ordered by `id` (assuming `id` is the incremental column)
            select_query = """
                    SELECT id, Water_Level, Food_Level
                    FROM sensordata4
                    ORDER BY id DESC
                    LIMIT 1
                """
            cursor.execute(select_query)
            results = cursor.fetchone()  # Fetch all rows

            print("Retrieved all data:", results)  # Debugging log
            return results
        except Error as e:
            print(f"Error: {e}")  # Log any errors
            return None
        finally:
            cursor.close()  # Close the cursor
            connection.close()  # Close the connection

@app.route('/get_data3')
def fetch_data3():
    data = get_data3()
    if data:
        # Return JSON with the necessary fields
        return jsonify({
            'Water_Level': data.get('Water_Level', 'N/A'),
            'Food_Level': data.get('Food_Level', 'N/A')
        })
    else:
        return jsonify({'error': 'No data found'})

# Manage Users

def get_all_users():
    connection = create_connection()  # Ensure this function establishes a DB connection
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)  # Return rows as dictionaries
            # Query to fetch all rows from the "users" table, ordered by ID (or another column as needed)
            select_query = """
                SELECT id, Email, Username, Password
                FROM users
                ORDER BY id ASC
            """
            cursor.execute(select_query)
            results = cursor.fetchall()  # Fetch all rows
            print("Retrieved all users:", results)  # Debugging log
            return results
        except Error as e:
            print(f"Error: {e}")  # Log any errors
            return None
        finally:
            cursor.close()  # Close the cursor
            connection.close()  # Close the connection

@app.route('/get_all_data6')
def fetch_all_data6():
    data = get_all_users()
    if data:  # Check if data exists
        # Return all rows as JSON
        return jsonify(data)  # Just return the data, Flask will automatically format it as JSON
    else:
        return jsonify({'error': 'No data found'})  # Handle case with no data

# ------------------- All Record of Database -------------------

# ------------------- 1. Environment Data -------------------

def get_environment_data():
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            select_query = """
                SELECT DateTime, Temperature, Humidity, Light1, Light2, Light3, Light4, ExhaustFan
                FROM sensordata
                ORDER BY DateTime DESC
            """
            cursor.execute(select_query)
            results = cursor.fetchall()
            for result in results:
                if result.get("DateTime"):
                    original_datetime = result["DateTime"]
                    formatted_datetime = datetime.strptime(
                        str(original_datetime), "%Y-%m-%d %H:%M:%S"
                    ).strftime("%Y-%m-%d %I:%M:%S %p")
                    result["DateTime"] = formatted_datetime
            print("Retrieved environment data:", results)
            return results
        except Error as e:
            print(f"Error: {e}")
            return None
        finally:
            cursor.close()
            connection.close()

@app.route('/get_environment_data')
def fetch_environment_data():
    data = get_environment_data()
    if data:
        return jsonify(data)
    else:
        return jsonify({'error': 'No data found'})

# ------------------- 2. Growth Data -------------------

def get_growth_data():
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            select_query = """
                SELECT DateTime, ChickNumber, Weight
                FROM sensordata3
                ORDER BY DateTime DESC
            """
            cursor.execute(select_query)
            results = cursor.fetchall()
            for result in results:
                if result.get("DateTime"):
                    original_datetime = result["DateTime"]
                    formatted_datetime = datetime.strptime(
                        str(original_datetime), "%Y-%m-%d %H:%M:%S"
                    ).strftime("%Y-%m-%d %I:%M:%S %p")
                    result["DateTime"] = formatted_datetime
            print("Retrieved growth data:", results)
            return results
        except Error as e:
            print(f"Error: {e}")
            return None
        finally:
            cursor.close()
            connection.close()

@app.route('/get_growth_data')
def fetch_growth_data():
    data = get_growth_data()
    if data:
        return jsonify(data)
    else:
        return jsonify({'error': 'No data found'})

# ------------------- 3. Sanitization Data -------------------

def get_sanitization_data():
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            select_query = """
                SELECT DateTime, Conveyor, Sprinkle, UVLight
                FROM sensordata2
                ORDER BY DateTime DESC
            """
            cursor.execute(select_query)
            results = cursor.fetchall()
            for result in results:
                if result.get("DateTime"):
                    original_datetime = result["DateTime"]
                    formatted_datetime = datetime.strptime(
                        str(original_datetime), "%Y-%m-%d %H:%M:%S"
                    ).strftime("%Y-%m-%d %I:%M:%S %p")
                    result["DateTime"] = formatted_datetime
            print("Retrieved sanitization data:", results)
            return results
        except Error as e:
            print(f"Error: {e}")
            return None
        finally:
            cursor.close()
            connection.close()

@app.route('/get_sanitization_data')
def fetch_sanitization_data():
    data = get_sanitization_data()
    if data:
        return jsonify(data)
    else:
        return jsonify({'error': 'No data found'})

# ------------------- 4. Supplies Data -------------------

def get_supplies_data():
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            select_query = """
                SELECT DateTime, Food, Water
                FROM sensordata1
                ORDER BY DateTime DESC
            """
            cursor.execute(select_query)
            results = cursor.fetchall()
            for result in results:
                if result.get("DateTime"):
                    original_datetime = result["DateTime"]
                    formatted_datetime = datetime.strptime(
                        str(original_datetime), "%Y-%m-%d %H:%M:%S"
                    ).strftime("%Y-%m-%d %I:%M:%S %p")
                    result["DateTime"] = formatted_datetime
            print("Retrieved supplies data:", results)
            return results
        except Error as e:
            print(f"Error: {e}")
            return None
        finally:
            cursor.close()
            connection.close()

@app.route('/get_supplies_data')
def fetch_supplies_data():
    data = get_supplies_data()
    if data:
        return jsonify(data)
    else:
        return jsonify({'error': 'No data found'})

# ------------------- 5. Notifications Data -------------------

def get_notifications_data():
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            select_query = """
                SELECT DateTime, message
                FROM notifications
                ORDER BY DateTime DESC
            """
            cursor.execute(select_query)
            results = cursor.fetchall()
            for result in results:
                if result.get("DateTime"):
                    original_datetime = result["DateTime"]
                    formatted_datetime = datetime.strptime(
                        str(original_datetime), "%Y-%m-%d %H:%M:%S"
                    ).strftime("%Y-%m-%d %I:%M:%S %p")
                    result["DateTime"] = formatted_datetime
            print("Retrieved notifications data:", results)
            return results
        except Error as e:
            print(f"Error: {e}")
            return None
        finally:
            cursor.close()
            connection.close()

@app.route('/get_notifications_data')
def fetch_notifications_data():
    data = get_notifications_data()
    if data:
        return jsonify(data)
    else:
        return jsonify({'error': 'No data found'})

def get_chickstatus_data():
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            # Removed LIMIT so all rows are returned
            select_query = """
                SELECT DateTime, status
                FROM chickstatus
                ORDER BY DateTime DESC
            """
            cursor.execute(select_query)
            results = cursor.fetchall()
            for result in results:
                if result.get("DateTime"):
                    original_datetime = result["DateTime"]
                    formatted_datetime = datetime.strptime(
                        str(original_datetime), "%Y-%m-%d %H:%M:%S"
                    ).strftime("%Y-%m-%d %I:%M:%S %p")
                    result["DateTime"] = formatted_datetime
            print("Retrieved chickstatus data:", results)
            return results
        except Error as e:
            print(f"Error: {e}")
            return None
        finally:
            cursor.close()
            connection.close()


@app.route('/get_chickstatus_data')
def fetch_chickstatus_data():
    data = get_chickstatus_data()
    if data:
        return jsonify(data)
    else:
        return jsonify({'error': 'No data found'})

# ------------------- Filtered Data -------------------

def get_notifications_data1(filter_type=None):
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)

            # Set the time range for filtering
            today = datetime.today()
            if filter_type == "daily":
                start_date = today.replace(hour=0, minute=0, second=0)
            elif filter_type == "weekly":
                start_date = today - timedelta(days=7)
            elif filter_type == "monthly":
                start_date = today - timedelta(days=30)
            else:
                start_date = None  # No filter (fetch all data)

            # SQL Query with Filtering
            if start_date:
                select_query = """
                    SELECT DateTime, message 
                    FROM notifications 
                    WHERE DateTime >= %s 
                    ORDER BY DateTime DESC
                """
                cursor.execute(select_query, (start_date,))
            else:
                select_query = "SELECT DateTime, message FROM notifications ORDER BY DateTime DESC"
                cursor.execute(select_query)

            results = cursor.fetchall()
            for result in results:
                if result.get("DateTime"):
                    original_datetime = result["DateTime"]
                    formatted_datetime = datetime.strptime(
                        str(original_datetime), "%Y-%m-%d %H:%M:%S"
                    ).strftime("%Y-%m-%d %I:%M:%S %p")
                    result["DateTime"] = formatted_datetime

            print("Retrieved notifications data:", results)
            return results
        except Error as e:
            print(f"Error: {e}")
            return None
        finally:
            cursor.close()
            connection.close()


@app.route('/get_notifications_data1')
def fetch_notifications_data1():
    filter_type = request.args.get("filter")  # Get filter type from URL parameter
    data = get_notifications_data1(filter_type)
    if data:
        return jsonify(data)
    else:
        return jsonify({'error': 'No data found'})


## ------------------- 1. Chick Status Data -------------------

def get_chickstatus_data1(filter_type=None):
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            today = datetime.today()
            start_date = None

            if filter_type == "daily":
                start_date = today.replace(hour=0, minute=0, second=0)
            elif filter_type == "weekly":
                start_date = today - timedelta(days=7)
            elif filter_type == "monthly":
                start_date = today - timedelta(days=30)

            if start_date:
                select_query = """
                    SELECT DateTime, ChickNumber, status 
                    FROM chickstatus 
                    WHERE DateTime >= %s 
                    ORDER BY DateTime DESC
                """
                cursor.execute(select_query, (start_date,))
            else:
                select_query = "SELECT DateTime, status FROM chickstatus ORDER BY DateTime DESC"
                cursor.execute(select_query)

            results = cursor.fetchall()
            for result in results:
                if result.get("DateTime"):
                    original_datetime = result["DateTime"]
                    formatted_datetime = datetime.strptime(
                        str(original_datetime), "%Y-%m-%d %H:%M:%S"
                    ).strftime("%Y-%m-%d %I:%M:%S %p")
                    result["DateTime"] = formatted_datetime

            print("Retrieved chickstatus data:", results)
            return results
        except Error as e:
            print(f"Error: {e}")
            return None
        finally:
            cursor.close()
            connection.close()

@app.route('/get_chickstatus_data1')
def fetch_chickstatus_data1():
    filter_type = request.args.get("filter")
    data = get_chickstatus_data1(filter_type)
    return jsonify(data) if data else jsonify({'error': 'No data found'})

# ------------------- 2. Environment Data -------------------

def get_environment_data1(filter_type=None):
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            today = datetime.today()
            start_date = None

            if filter_type == "daily":
                start_date = today.replace(hour=0, minute=0, second=0)
            elif filter_type == "weekly":
                start_date = today - timedelta(days=7)
            elif filter_type == "monthly":
                start_date = today - timedelta(days=30)

            if start_date:
                select_query = """
                    SELECT DateTime, Temperature, Humidity, Light1, Light2, Light3, Light4, ExhaustFan
                    FROM sensordata 
                    WHERE DateTime >= %s 
                    ORDER BY DateTime DESC
                """
                cursor.execute(select_query, (start_date,))
            else:
                select_query = "SELECT DateTime, Temperature, Humidity, Light1, Light2, Light3, Light4, ExhaustFan FROM sensordata ORDER BY DateTime DESC"
                cursor.execute(select_query)

            results = cursor.fetchall()
            for result in results:
                if result.get("DateTime"):
                    original_datetime = result["DateTime"]
                    formatted_datetime = datetime.strptime(
                        str(original_datetime), "%Y-%m-%d %H:%M:%S"
                    ).strftime("%Y-%m-%d %I:%M:%S %p")
                    result["DateTime"] = formatted_datetime

            print("Retrieved environment data:", results)
            return results
        except Error as e:
            print(f"Error: {e}")
            return None
        finally:
            cursor.close()
            connection.close()

@app.route('/get_environment_data1')
def fetch_environment_data1():
    filter_type = request.args.get("filter")
    data = get_environment_data1(filter_type)
    return jsonify(data) if data else jsonify({'error': 'No data found'})

# ------------------- 3. Growth Data -------------------

def get_growth_data1(filter_type=None):
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            today = datetime.today()
            start_date = None

            if filter_type == "daily":
                start_date = today.replace(hour=0, minute=0, second=0)
            elif filter_type == "weekly":
                start_date = today - timedelta(days=7)
            elif filter_type == "monthly":
                start_date = today - timedelta(days=30)

            if start_date:
                select_query = """
                    SELECT DateTime, ChickNumber, Weight 
                    FROM sensordata3 
                    WHERE DateTime >= %s 
                    ORDER BY DateTime DESC
                """
                cursor.execute(select_query, (start_date,))
            else:
                select_query = "SELECT DateTime, Weight FROM sensordata3 ORDER BY DateTime DESC"
                cursor.execute(select_query)

            results = cursor.fetchall()
            for result in results:
                if result.get("DateTime"):
                    original_datetime = result["DateTime"]
                    formatted_datetime = datetime.strptime(
                        str(original_datetime), "%Y-%m-%d %H:%M:%S"
                    ).strftime("%Y-%m-%d %I:%M:%S %p")
                    result["DateTime"] = formatted_datetime

            print("Retrieved growth data:", results)
            return results
        except Error as e:
            print(f"Error: {e}")
            return None
        finally:
            cursor.close()
            connection.close()

@app.route('/get_growth_data1')
def fetch_growth_data1():
    filter_type = request.args.get("filter")
    data = get_growth_data1(filter_type)
    return jsonify(data) if data else jsonify({'error': 'No data found'})

# ------------------- 4. Sanitization Data -------------------

def get_sanitization_data1(filter_type=None):
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            today = datetime.today()
            start_date = None

            if filter_type == "daily":
                start_date = today.replace(hour=0, minute=0, second=0)
            elif filter_type == "weekly":
                start_date = today - timedelta(days=7)
            elif filter_type == "monthly":
                start_date = today - timedelta(days=30)

            if start_date:
                select_query = """
                    SELECT DateTime, Conveyor, Sprinkle, UVLight 
                    FROM sensordata2 
                    WHERE DateTime >= %s 
                    ORDER BY DateTime DESC
                """
                cursor.execute(select_query, (start_date,))
            else:
                select_query = "SELECT DateTime, Conveyor, Sprinkle, UVLight FROM sensordata2 ORDER BY DateTime DESC"
                cursor.execute(select_query)

            results = cursor.fetchall()
            for result in results:
                if result.get("DateTime"):
                    original_datetime = result["DateTime"]
                    formatted_datetime = datetime.strptime(
                        str(original_datetime), "%Y-%m-%d %H:%M:%S"
                    ).strftime("%Y-%m-%d %I:%M:%S %p")
                    result["DateTime"] = formatted_datetime

            print("Retrieved sanitization data:", results)
            return results
        except Error as e:
            print(f"Error: {e}")
            return None
        finally:
            cursor.close()
            connection.close()

@app.route('/get_sanitization_data1')
def fetch_sanitization_data1():
    filter_type = request.args.get("filter")
    data = get_sanitization_data1(filter_type)
    return jsonify(data) if data else jsonify({'error': 'No data found'})

# ------------------- 4. Supplies Data -------------------

def get_supplies_data1(filter_type=None):
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor(dictionary=True)
            today = datetime.today()
            start_date = None

            if filter_type == "daily":
                start_date = today.replace(hour=0, minute=0, second=0)
            elif filter_type == "weekly":
                start_date = today - timedelta(days=7)
            elif filter_type == "monthly":
                start_date = today - timedelta(days=30)

            if start_date:
                select_query = """
                    SELECT DateTime, Food, Water 
                    FROM sensordata1 
                    WHERE DateTime >= %s 
                    ORDER BY DateTime DESC
                """
                cursor.execute(select_query, (start_date,))
            else:
                select_query = "SELECT DateTime, Food, Water FROM sensordata1 ORDER BY DateTime DESC"
                cursor.execute(select_query)

            results = cursor.fetchall()
            for result in results:
                if result.get("DateTime"):
                    original_datetime = result["DateTime"]
                    formatted_datetime = datetime.strptime(
                        str(original_datetime), "%Y-%m-%d %H:%M:%S"
                    ).strftime("%Y-%m-%d %I:%M:%S %p")
                    result["DateTime"] = formatted_datetime

            print("Retrieved supplies data:", results)
            return results
        except Error as e:
            print(f"Error: {e}")
            return None
        finally:
            cursor.close()
            connection.close()

@app.route('/get_supplies_data1')
def fetch_supplies_data1():
    filter_type = request.args.get("filter")
    data = get_supplies_data1(filter_type)
    return jsonify(data) if data else jsonify({'error': 'No data found'})


@app.route('/update_user', methods=['POST'])
def update_user():
    data = request.json
    user_id = data.get("id")
    new_email = data.get("email")
    new_username = data.get("username")
    new_password = data.get("password")

    if not user_id:
        return jsonify({"success": False, "message": "User ID is required"}), 400

    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor()
            update_query = """
                UPDATE users
                SET Email = %s, Username = %s, Password = %s
                WHERE id = %s
            """
            cursor.execute(update_query, (new_email, new_username, new_password, user_id))
            connection.commit()
            return jsonify({"success": True})
        except Error as e:
            print(f"Error updating user: {e}")
            return jsonify({"success": False, "message": str(e)}), 500
        finally:
            cursor.close()
            connection.close()

    return jsonify({"success": False, "message": "Database connection failed"}), 500

@app.route('/delete_user', methods=['POST'])
def delete_user():
    data = request.json
    user_id = data.get("id")

    if not user_id:
        return jsonify({"success": False, "message": "User ID is required"}), 400

    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor()
            delete_query = "DELETE FROM users WHERE id = %s"
            cursor.execute(delete_query, (user_id,))
            connection.commit()
            return jsonify({"success": True})
        except Error as e:
            print(f"Error deleting user: {e}")
            return jsonify({"success": False, "message": str(e)}), 500
        finally:
            cursor.close()
            connection.close()

    return jsonify({"success": False, "message": "Database connection failed"}), 500

# Report Search and Filter

@app.route('/get_filtered_data')
def get_filtered_data():
    record_type = request.args.get("recordType")  # Type of data to search in
    from_date = request.args.get("fromDate")  # Date range start
    to_date = request.args.get("toDate")  # Date range end
    search_query = request.args.get("search")  # Search keyword

    connection = create_connection()
    if not connection:
        return jsonify({'error': 'Database connection failed'})

    try:
        cursor = connection.cursor(dictionary=True)

        # Table Mapping
        table_map = {
            "notifications": "notifications",
            "supplies": "sensordata1",
            "environment": "sensordata",
            "growth1": "sensordata3",
            "growth": "chickstatus",
            "sanitization": "sensordata2"
        }
        table_name = table_map.get(record_type, "notifications")  # Default to notifications

        # Get column names dynamically from the table
        cursor.execute(f"SHOW COLUMNS FROM {table_name}")
        columns = [col["Field"] for col in cursor.fetchall() if col["Field"] != "id"]  # Exclude 'id' column

        # Construct SQL Query
        query = f"SELECT * FROM {table_name} WHERE 1=1"
        params = []

        # ✅ Fix Date Range Format
        if from_date and to_date:
            try:
                from_date = datetime.strptime(from_date, "%Y-%m-%d").strftime("%Y-%m-%d 00:00:00")
                to_date = datetime.strptime(to_date, "%Y-%m-%d").strftime("%Y-%m-%d 23:59:59")
                query += " AND DateTime BETWEEN %s AND %s"
                params.extend([from_date, to_date])
            except ValueError:
                return jsonify({'error': 'Invalid date format'})

        # ✅ Make Search Query Work for All Columns
        if search_query:
            query += " AND ("
            query += " OR ".join([f"{col} LIKE %s" for col in columns])  # Search in all columns
            query += ")"
            params.extend([f"%{search_query}%"] * len(columns))  # Apply search to all columns

        # ✅ Order by newest data
        query += " ORDER BY DateTime DESC"

        # ✅ Execute Query
        cursor.execute(query, params)
        results = cursor.fetchall()

        # ✅ Format DateTime correctly
        for result in results:
            if result.get("DateTime"):
                original_datetime = result["DateTime"]
                formatted_datetime = datetime.strptime(str(original_datetime), "%Y-%m-%d %H:%M:%S").strftime(
                    "%Y-%m-%d %I:%M:%S %p")
                result["DateTime"] = formatted_datetime

        return jsonify(results) if results else jsonify({'error': 'No data found'})

    except Error as e:
        print(f"Error: {e}")
        return jsonify({'error': 'Query failed'})
    finally:
        cursor.close()
        connection.close()

SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_ADDRESS = "2020-200423@rtu.edu.ph"
EMAIL_PASSWORD = "otfu klum scxj akhz"  # App Password, not actual password

# Store last sent notification timestamp
last_sent_time = 0
EMAIL_COOLDOWN = 300  # 5 minutes cooldown

def send_email_notifications(user_email):
    global last_sent_time
    current_time = time.time()

    # Prevent sending emails too frequently
    if current_time - last_sent_time < EMAIL_COOLDOWN:
        print("Email cooldown active. Skipping email.")
        return

    try:
        db = mysql.connector.connect(
            host="127.0.0.1",
            user="root",
            password="",
            database="test"
        )
        cursor = db.cursor()

        # Fetch the **latest** notification
        cursor.execute("SELECT message FROM notifications ORDER BY DateTime DESC LIMIT 1")
        latest_notification = cursor.fetchone()

        if not latest_notification:
            print(f"No new notifications for {user_email}.")
            return

        notification_message = latest_notification[0]

        # Create Email Content
        msg = MIMEMultipart()
        msg["From"] = EMAIL_ADDRESS
        msg["To"] = user_email
        msg["Subject"] = "⚠️ Chick Care System Alert"

        body = f"""
        Dear User,

        🐥 **Chick Care Notification** 🐥

        {notification_message}

        Stay updated with your Chick Care system.
        """
        msg.attach(MIMEText(body, "plain"))

        # Send Email
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
        server.sendmail(EMAIL_ADDRESS, user_email, msg.as_string())
        server.quit()

        # Update last sent time
        last_sent_time = time.time()
        print(f"✅ Email sent to {user_email}: {notification_message}")

    except Exception as e:
        print(f"❌ Error sending email: {e}")

    finally:
        if db.is_connected():
            cursor.close()
            db.close()

@app.route('/trigger_email_alert', methods=['POST'])
def trigger_email_alert():
    try:
        # Get the logged-in user's email
        user_email = session.get("user_email")  # Ensure you store user email in session on login

        if user_email:
            send_email_notifications(user_email)
            return jsonify({"message": "Email sent successfully"}), 200
        else:
            return jsonify({"error": "User email not found"}), 400

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def get_text_from_url(url):
    try:
        response = requests.get(url, timeout=5)  # Fetch the webpage
        soup = BeautifulSoup(response.text, 'html.parser')
        extracted_text = soup.get_text().strip()  # Extract text content
        return extracted_text
    except Exception as e:
        print(f"Error fetching data from URL: {e}")
        return "Unknown"

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


def insert_message_to_mysql(status, ChickNumber):
    try:
        connection = mysql.connector.connect(host='127.0.0.1', user='root', password='', database='test')
        cursor = connection.cursor()
        query = "INSERT INTO chickstatus (ChickNumber, status) VALUES (%s, %s)"
        cursor.execute(query, (ChickNumber, status))

        # Insert notification message into notifications table
        notification_message = f"Poultry health detected: {ChickNumber} | {status}"
        insert_notification_query = """
                    INSERT INTO notifications (message, DateTime)
                    VALUES (%s, NOW())
                    """
        cursor.execute(insert_notification_query, (notification_message,))

        # Commit the transaction
        connection.commit()
        print(f"Message inserted: {status} and Notification sent.")
        print(f"Inserted: {status} | QR: {ChickNumber}")
    except mysql.connector.Error as err:
        print(f"MySQL Error: {err}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


HARVEST_DAYS = 30  # Number of days until harvest


def register_chick(ChickNumber):
    """ Register chick in database if not already registered """
    try:
        connection = mysql.connector.connect(host='127.0.0.1', user='root', password='', database='test')
        cursor = connection.cursor()

        # Check if chick is already registered
        cursor.execute("SELECT ChickNumber FROM chick_records WHERE ChickNumber = %s", (ChickNumber,))
        existing_chick = cursor.fetchone()

        if not existing_chick:
            # Insert new chick with current date
            today = datetime.now().date()
            query = "INSERT INTO chick_records (ChickNumber, registration_date) VALUES (%s, %s)"
            cursor.execute(query, (ChickNumber, today))
            connection.commit()
            print(f"Chick {ChickNumber} registered on {today}")
        else:
            print(f"Chick {ChickNumber} is already registered.")

    except mysql.connector.Error as err:
        print(f"MySQL Error: {err}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()


def check_harvest_time():
    """ Check if any chicks have reached harvest time """
    try:
        connection = mysql.connector.connect(host='127.0.0.1', user='root', password='', database='test')
        cursor = connection.cursor()

        # Get today's date
        today = datetime.now().date()

        # Find all chicks that have reached 30 days since registration
        query = """
        SELECT ChickNumber, registration_date FROM chick_records 
        WHERE DATE_ADD(registration_date, INTERVAL %s DAY) <= %s
        """
        cursor.execute(query, (HARVEST_DAYS, today))
        harvest_chicks = cursor.fetchall()

        for chick in harvest_chicks:
            ChickNumber, reg_date = chick
            print(f"⚠️ Chick {ChickNumber} is ready for harvest! (Registered on {reg_date})")

        if not harvest_chicks:
            print("✅ No chicks are ready for harvest yet.")

    except mysql.connector.Error as err:
        print(f"MySQL Error: {err}")
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

# Insert weight & chick data into MySQL
def insert_weight_data(weight_key):
    connection = create_connection()
    if connection:
        try:
            cursor = connection.cursor()

            weight = arduino_data[weight_key]
            if weight >= WEIGHT_THRESHOLD:  # Only process if weight is above threshold
                chick_number = Chick_Number  # Scan QR code

                if chick_number:  # Insert only if both weight and ChickNumber exist
                    insert_query = """
                        INSERT INTO sensordata3 (DateTime, ChickNumber, Weight)
                        VALUES (NOW(), %s, %s)
                    """
                    cursor.execute(insert_query, (chick_number, weight))
                    connection.commit()
                    print(f"Data inserted for {weight_key}: Chick {chick_number}, Weight {weight}")
                else:
                    print(f"Weight detected on {weight_key}, but no chick QR code scanned.")


            else:
                print(f"No chick detected on {weight_key} (weight below threshold: {weight} kg)")

        except mysql.connector.Error as e:
            print(f"MySQL Error: {e}")
        finally:
            cursor.close()
            connection.close()









if __name__ == '__main__':
    app.run(debug=True)

