from flask import Flask, render_template, request, redirect, url_for, send_file
import os
import cv2
import numpy as np
import tensorflow.lite as tflite
import google.generativeai as genai
from reportlab.pdfgen import canvas
import cv2
import numpy as np
import tensorflow.lite as tflite
import tkinter as tk
from tkinter import filedialog

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Configure Google Gemini API
genai.configure(api_key=" Gemini API")

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="plant_disease_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# Define labels
labels = ['Apple Scab', 'Apple Cedar Rust', 'Apple Leaf Spot', 'Apple Powdery Mildew', 'Unknown','Apple Fruit Rot',
    'Tomato Early Blight', 'Tomato Late Blight', 'Tomato Septoria Leaf Spot', 'Tomato Fusarium Wilt', 
    'Tomato Verticillium Wilt', 'Potato Late Blight', 'Potato Early Blight', 'Potato Scab', 'Corn Rust', 
    'Corn Blight', 'Corn Smut', 'Wheat Rust', 'Wheat Blight', 'Wheat Powdery Mildew', 'Pepper Bacterial Spot', 
    'Pepper Powdery Mildew', 'Strawberry Leaf Spot', 'Strawberry Powdery Mildew', 'Strawberry Botrytis Fruit Rot', 
    'Squash Blossom End Rot', 'Cabbage Worm', 'Cabbage Downy Mildew', 'Cabbage Black Rot', 'Tomato Spider Mites', 
    'Tomato Leaf Mold', 'Tomato Healthy', 'Apple Black Rot', 'Apple Fire Blight', 'Grape Black Rot', 'Grape Healthy', 
    'Peach Bacterial Spot', 'Peach Healthy', 'Soybean Rust', 'Squash Mosaic Virus', 'Rice Blast', 'Rice Sheath Blight', 
    'Rice Brown Spot', 'Rice Healthy', 'Citrus Greening', 'Citrus Healthy', 'Mango Anthracnose', 'Mango Healthy', 
    'Cotton Wilt', 'Cotton Healthy', 'Banana Black Sigatoka', 'Banana Healthy', 'Coffee Leaf Rust', 'Coffee Healthy', 
    'Pear Leaf Spot', 'Pear Fire Blight', 'Pear Healthy', 'Pomegranate Bacterial Spot', 'Pomegranate Healthy', 
    'Guava Wilt', 'Guava Healthy', 'Lettuce Downy Mildew', 'Lettuce Healthy', 'Spinach Leaf Spot', 'Spinach Healthy', 
    'Brinjal Wilt', 'Brinjal Healthy', 'Okra Yellow Vein Mosaic Virus', 'Okra Healthy', 'Zucchini Mosaic Virus', 
    'Zucchini Healthy', 'Turnip Leaf Spot', 'Turnip Healthy', 'Mustard Leaf Spot', 'Mustard Healthy', 'Kale Healthy', 
    'Tomato Blossom End Rot', 'Tomato Bacterial Wilt', 'Tomato Anthracnose', 'Tomato White Mold', 'Tomato Target Spot']  

# Define disease prevention advice




def get_gemini_advice(disease):
    model = genai.GenerativeModel("gemini-pro")
    prompt = f"Provide detailed prevention and treatment advice for {disease} in plants."

    try:
        response = model.generate_content(prompt)
        return response.text.strip() if response and hasattr(response, 'text') else "No advice available."
    except Exception as e:
        return f"Error fetching advice: {str(e)}"

def preprocess_image(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img = cv2.resize(img, (input_shape[1], input_shape[2]))
    img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0
    return img

def predict_disease(image_path):
    img = preprocess_image(image_path)
    if img is None:
        return "Error", 0.0, "No image found."

    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])[0]
    class_id = np.argmax(predictions)
    confidence = predictions[class_id]
    disease_name = labels[class_id] if class_id < len(labels) else "Unknown"
    prevention_advice = advice_dict.get(disease_name, get_gemini_advice(disease_name))
    return disease_name, confidence, prevention_advice

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
@app.route('/upload', methods=['POST'])
@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Load and preprocess image
            image = cv2.imread(filepath)
            image = cv2.resize(image, (input_shape[1], input_shape[2]))
            image = np.expand_dims(image, axis=0).astype(np.float32) / 255.0

            # Run inference
            interpreter.set_tensor(input_details[0]['index'], image)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            # Get prediction
            prediction_index = np.argmax(output_data)
            prediction = labels[prediction_index]
            confidence = float(output_data[0][prediction_index])
            advice = advice_dict.get(prediction, "No specific advice available.")

            return render_template('upload.html', filename=filename, prediction=prediction, confidence=confidence, advice=advice)
    
    return render_template('upload.html')


@app.route('/download_report/<filename>/<prediction>/<confidence>/<advice>')
def download_report(filename, prediction, confidence, advice):
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], "report.pdf")
    image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

    c = canvas.Canvas(pdf_path)
    c.setFont("Helvetica", 16)
    
    # Title
    c.drawString(200, 800, "🌱 Plant Disease Detection Report")
    c.line(50, 790, 550, 790)  # Horizontal line
    print(end="\n\n")
    
    # Add Image
    try:
        print(end="\n")
        c.drawImage(image_path, 180, 500, width=250, height=170)  # Adjusted image position and size
    except Exception as e:
        c.drawString(100, 610, "Image could not be loaded.")

    # Disease Prediction
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, 460, f"🦠 Disease Detected: {prediction}")

    # Confidence Score
    c.setFont("Helvetica", 12)
    c.drawString(100, 440, f"Confidence Score: {float(confidence):.2f}")

    # Prevention Advice Title
    c.setFont("Helvetica-Bold", 14)
    c.drawString(100, 420, "----> Prevention & Treatment Advice:")

    # Prevention Advice Content
    c.setFont("Helvetica", 12)
    y = 400
    for line in advice.split('. '):
        c.drawString(120, y, f"- {line.strip()}.")
        y -= 20

    c.save()
    
    return send_file(pdf_path, as_attachment=True)

@app.route('/live')
def live_page():
    return render_template('live.html')
@app.route('/live', methods=['GET'])
def live():
    return render_template('live.html')  # Render a new template for live detection.

def start_live_detection():
        # Get input and output tensors
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

# Get the expected input shape
    input_shape = input_details[0]['shape'] 


    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
    
    # Apply Gaussian blur to reduce noise
        frame_blur = cv2.GaussianBlur(frame, (5, 5), 0)
    
    # Convert to grayscale and resize to model input size
        img = cv2.cvtColor(frame_blur, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (input_shape[1], input_shape[2]))  

    # Normalize input
        img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0  

    # Run inference
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
    
    # Get prediction results
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        predictions = np.exp(predictions) / np.sum(np.exp(predictions))  # Softmax Scaling

    # Get highest confidence class
        class_id = np.argmax(predictions)
        confidence = predictions[class_id]

    # Validate class_id
        label = labels[class_id] if class_id < len(labels) else "Unknown"
    
    # Get frame dimensions
        h, w, _ = frame.shape
        startX, startY, endX, endY = int(w * 0.15), int(h * 0.15), int(w * 0.75), int(h * 0.75)

    # Define color based on confidence scoreq
        color = (0, 255, 0) if confidence > 0.75 else (0, 0, 255)

    # Draw bounding box
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 1)
        cv2.putText(frame, f"Disease : {label}  (Acc: {confidence:.2f})", 
                    (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)
    
    # Display output
        cv2.imshow("Plant Disease Detection", frame)

    # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
if __name__ == '__main__':
    app.run(debug=True)
