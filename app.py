from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file, Response
from flask_mail import Mail, Message
import os
import cv2
import numpy as np
import tensorflow.lite as tflite
import google.generativeai as genai
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from datetime import datetime
import logging
from dotenv import load_dotenv
import requests
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from flask import Flask, send_file
import os
import requests
from datetime import datetime
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfbase import pdfmetrics
from fpdf import FPDF

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configuration
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, send_file, Response
from flask_mail import Mail, Message
import os
import cv2
import numpy as np
import tensorflow.lite as tflite
import google.generativeai as genai
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from datetime import datetime
import logging
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Configuration
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['VOICE_FOLDER'] = 'voice/'
app.config['DOWNLOAD_FOLDER'] = 'static/pdf_reports'
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key')
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = os.getenv('MAIL_USERNAME', 'muthumanirajs3@gmail.com')
app.config['MAIL_PASSWORD'] = os.getenv('MAIL_PASSWORD', 'tntb auaw zkcz pnsg')
app.config['MAIL_DEFAULT_SENDER'] = os.getenv('MAIL_USERNAME', 'muthumanirajs3@gmail.com')
# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['VOICE_FOLDER'], exist_ok=True)
os.makedirs(app.config['DOWNLOAD_FOLDER'], exist_ok=True)

# Initialize Flask-Mail
mail = Mail(app)

# Load TFLite model
interpreter = tflite.Interpreter(model_path='new_plant_disease_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# Load labels
labels = [
    "Apple scab", "Apple black rot", "Apple cedar apple rust", "Apple healthy",
    "Background without leaves", "Blueberry healthy", "Cherry powdery mildew",
    "Cherry healthy", "Corn gray leaf spot", "Corn common rust",
    "Corn northern leaf blight", "Corn healthy", "Grape black rot",
    "Grape black measles", "Grape leaf blight", "Grape healthy",
    "Orange haunglongbing", "Peach bacterial spot", "Peach healthy",
    "Pepper bacterial spot", "Pepper healthy", "Potato early blight",
    "Potato healthy", "Potato late blight", "Raspberry healthy",
    "Soybean healthy", "Squash powdery mildew", "Strawberry healthy",
    "Strawberry leaf scorch", "Tomato bacterial spot", "Tomato early blight",
    "Tomato healthy", "Tomato late blight", "Tomato leaf mold",
    "Tomato septoria leaf spot", "Tomato spider mites two spotted spider mite",
    "Tomato target spot", "Tomato mosaic virus", "Tomato yellow leaf curl virus",
    "Unknown"
]

# Initialize Google Gemini API
genai.configure(api_key="API")
model = genai.GenerativeModel("gemini-1.5-pro")

# Helper Functions
def get_gemini_advice(disease_name):
    """Get prevention and treatment advice from Gemini API."""
    prompt = f"Provide detailed prevention and treatment recommendations for {disease_name} in plants."
    response = model.generate_content(prompt)
    return response.text if response else "No advice available."

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        email = request.form['email']
        subject = request.form['subject']
        message = request.form['message']

        msg = Message(subject, recipients=[app.config['MAIL_USERNAME']], body=f"Message from: {email}\n\n{message}")
        try:
            mail.send(msg)
            flash('Your message has been sent successfully!', 'success')
        except Exception as e:
            flash(f'An error occurred: {str(e)}', 'error')

        return redirect(url_for('contact'))

    return render_template('contact.html')

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
            image = cv2.resize(image, (224, 224))  # Adjust input size as per your model
            image = np.expand_dims(image, axis=0).astype(np.float32) / 255.0

            # Run inference
            interpreter.set_tensor(input_details[0]['index'], image)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            # Get prediction
            prediction_index = np.argmax(output_data)
            prediction = labels[prediction_index]
            confidence = float(output_data[0][prediction_index])

            # Get advice
            advice = get_gemini_advice(prediction)
            advice = advice.replace("*", "").replace("✔ ", "").replace("#", "")

            return render_template(
                'upload.html',
                filename=filename,
                prediction=prediction,
                confidence=confidence,
                advice=advice
            )

    return render_template('upload.html')

# @app.route('/download_pdf/<prediction>/<confidence>/<advice>/<language>')
# def download_pdf(prediction, confidence, advice, language):
#     # Translate the advice text if the language is not English
#     if language != "en":
#         try:
#             response = requests.get(
#                 f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl={language}&dt=t&q={advice}"
#             )
#             translated_text = " ".join([item[0] for item in response.json()[0]])
#             advice = translated_text
#         except Exception as e:
#             print(f"Translation error: {e}")

#     # Create PDF document
#     pdf_filename = "Plant_Disease_Report.pdf"
#     pdf_path = os.path.join(app.config['DOWNLOAD_FOLDER'], pdf_filename)

#     doc = SimpleDocTemplate(pdf_path, pagesize=letter)
#     styles = getSampleStyleSheet()
#     story = []

#     # Title
#     title = Paragraph("<b><font size=18>🌿 Plant Disease Detection Report</font></b>", styles["Title"])
#     story.append(title)
#     story.append(Spacer(1, 12))

#     # Disease & Confidence Score Table
#     table_data = [
#         ["Disease Detected:", prediction],
#         ["Confidence Score:", f"{round(float(confidence), 2) * 100}%"]
#     ]
#     table = Table(table_data, colWidths=[150, 350])
#     table.setStyle(TableStyle([
#         ('BACKGROUND', (0, 0), (-1, 0), colors.lightgreen),
#         ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
#         ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
#         ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
#         ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
#         ('GRID', (0, 0), (-1, -1), 1, colors.green)
#     ]))
#     story.append(table)
#     story.append(Spacer(1, 20))

#     # Advice Section
#     story.append(Paragraph("<b>🌱 Prevention & Treatment Advice:</b>", styles["Heading2"]))
#     story.append(Spacer(1, 10))

#     for tip in advice.split('. '):
#         if tip.strip():
#             story.append(Paragraph(f" {tip.strip()}.", styles["Normal"]))
#             story.append(Spacer(1, 5))

#     # Footer with Timestamp
#     timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#     story.append(Spacer(1, 20))
#     story.append(Paragraph(f"<i>Generated on: {timestamp}</i>", ParagraphStyle(name="Italic", fontSize=10, textColor=colors.grey)))

#     # Build PDF
#     doc.build(story)

#     return send_file(pdf_path, as_attachment=True)
pdfmetrics.registerFont(TTFont('NotoSans', 'styles/NotoSans-Regular.ttf'))

@app.route('/download_pdf/<prediction>/<confidence>/<advice>/<language>')
def download_pdf(prediction, confidence, advice, language):
    # Translate advice text if needed
    if language != "en":
        try:
            response = requests.get(
                f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl={language}&dt=t&q={advice}"
            )
            translated_text = " ".join([item[0] for item in response.json()[0]])
            advice = translated_text
        except Exception as e:
            print(f"Translation error: {e}")

    # Create PDF document
    pdf_filename = "Plant_Disease_Report.pdf"
    pdf_path = os.path.join(app.config['DOWNLOAD_FOLDER'], pdf_filename)

    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Use a language-compatible font
    normal_style = ParagraphStyle(name="Normal", fontName="NotoSans", fontSize=12)
    
    # Title
    title = Paragraph("<b><font size=18>🌿 Plant Disease Detection Report</font></b>", styles["Title"])
    story.append(title)
    story.append(Spacer(1, 12))

    # Disease & Confidence Score Table
    table_data = [
        ["Disease Detected:", prediction],
        ["Confidence Score:", f"{round(float(confidence), 2) * 100}%"]
    ]
    table = Table(table_data, colWidths=[150, 350])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.lightgreen),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 1, colors.green)
    ]))
    story.append(table)
    story.append(Spacer(1, 20))

    # Advice Section
    story.append(Paragraph("<b>🌱 Prevention & Treatment Advice:</b>", styles["Heading2"]))
    story.append(Spacer(1, 10))

    for tip in advice.split('. '):
        if tip.strip():
            story.append(Paragraph(f" {tip.strip()}.", normal_style))
            story.append(Spacer(1, 5))

    # Footer with Timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    story.append(Spacer(1, 20))
    story.append(Paragraph(f"<i>Generated on: {timestamp}</i>", ParagraphStyle(name="Italic", fontSize=10, textColor=colors.grey)))

    # Build PDF
    doc.build(story)

    return send_file(pdf_path, as_attachment=True)

@app.route('/live')
def live_page():
    return render_template('live.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess and predict
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0

        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        class_id = np.argmax(predictions)
        confidence = predictions[class_id]
        label = labels[class_id] if class_id < len(labels) else "Unknown"

        # Draw results on frame
        cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

if __name__ == '__main__':
    app.run(debug=True)

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['VOICE_FOLDER'], exist_ok=True)
os.makedirs(app.config['DOWNLOAD_FOLDER'], exist_ok=True)

# Initialize Flask-Mail
mail = Mail(app)

# Load TFLite model
interpreter = tflite.Interpreter(model_path='new_plant_disease_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape']

# Load labels
labels = [
    "Apple scab", "Apple black rot", "Apple cedar apple rust", "Apple healthy",
    "Background without leaves", "Blueberry healthy", "Cherry powdery mildew",
    "Cherry healthy", "Corn gray leaf spot", "Corn common rust",
    "Corn northern leaf blight", "Corn healthy", "Grape black rot",
    "Grape black measles", "Grape leaf blight", "Grape healthy",
    "Orange haunglongbing", "Peach bacterial spot", "Peach healthy",
    "Pepper bacterial spot", "Pepper healthy", "Potato early blight",
    "Potato healthy", "Potato late blight", "Raspberry healthy",
    "Soybean healthy", "Squash powdery mildew", "Strawberry healthy",
    "Strawberry leaf scorch", "Tomato bacterial spot", "Tomato early blight",
    "Tomato healthy", "Tomato late blight", "Tomato leaf mold",
    "Tomato septoria leaf spot", "Tomato spider mites two spotted spider mite",
    "Tomato target spot", "Tomato mosaic virus", "Tomato yellow leaf curl virus",
    "Unknown"
]

# Initialize Google Gemini API
genai.configure(api_key="AIzaSyAzKeQdOqZJo2Hmi2NpZ34jHDBgppkxwGU")
model = genai.GenerativeModel("gemini-2.0-flash")

# Register Unicode fonts using absolute paths
font_dir = os.path.abspath('static/fonts')

# Debug: Print the font directory
print(f"Font Directory: {font_dir}")

# Register fonts with debugging
fonts = {
    'NotoSans': 'NotoSans-Regular.ttf',
    'NotoSansHindi': 'NotoSansHindi-Regular.ttf',
    'NotoSansTelugu': 'NotoSansTelugu-Regular.ttf',
    'NotoSansMalayalam': 'NotoSansMalayalam-Regular.ttf',
    'NotoSansKannada': 'NotoSansKannada-Regular.ttf',
    'NotoSansGurmukhi': 'NotoSansGurmukhi-Regular.ttf',
    'NotoSansMarathi': 'NotoSansMarathi-Regular.ttf',
    'NotoSansBengali': 'NotoSansBengali-Regular.ttf',
    'NotoSansGujarati': 'NotoSansGujarati-Regular.ttf'
}

for font_name, font_file in fonts.items():
    font_path = os.path.join(font_dir, font_file)
    print(f"Registering Font: {font_name} -> {font_path}")  # Debug statement
    try:
        pdfmetrics.registerFont(TTFont(font_name, font_path))
    except Exception as e:
        print(f"Error registering font {font_name}: {e}")

# Verify registered fonts
print("Registered Fonts:", pdfmetrics.getRegisteredFontNames())

# Helper Functions
def get_gemini_advice(disease_name):
    """Get prevention and treatment advice from Gemini API."""
    prompt = f"Provide detailed prevention and treatment recommendations for {disease_name} in plants."
    response = model.generate_content(prompt)
    return response.text if response else "No advice available."

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        email = request.form['email']
        subject = request.form['subject']
        message = request.form['message']

        # Send email to admin (you)
        try:
            msg = Message(
                subject=subject,
                recipients=[app.config['MAIL_USERNAME']],  # Send to your email
                body=f"Message from: {email}\n\n{message}",
                sender=app.config['MAIL_DEFAULT_SENDER']
            )
            mail.send(msg)
            flash('Your message has been sent successfully!', 'success')
        except Exception as e:
            flash(f'An error occurred: {str(e)}', 'error')

        return redirect(url_for('contact'))

    return render_template('contact.html')

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
            image = cv2.resize(image, (224, 224))  # Adjust input size as per your model
            image = np.expand_dims(image, axis=0).astype(np.float32) / 255.0

            # Run inference
            interpreter.set_tensor(input_details[0]['index'], image)
            interpreter.invoke()
            output_data = interpreter.get_tensor(output_details[0]['index'])

            # Get prediction
            prediction_index = np.argmax(output_data)
            prediction = labels[prediction_index]
            confidence = float(output_data[0][prediction_index])

            # Get advice
            advice = get_gemini_advice(prediction)
            advice = advice.replace("*", "").replace("✔ ", "").replace("#", "")

            return render_template(
                'upload.html',
                filename=filename,
                prediction=prediction,
                confidence=confidence,
                advice=advice
            )

    return render_template('upload.html')

@app.route('/download_pdf/<prediction>/<confidence>/<advice>/<language>')
def download_pdf(prediction, confidence, advice, language):
    # Translate the advice text if the language is not English
    if language != "en":
        try:
            response = requests.get(
                f"https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl={language}&dt=t&q={advice}"
            )
            translated_text = " ".join([item[0] for item in response.json()[0]])
            advice = translated_text
        except Exception as e:
            print(f"Translation error: {e}")

    # Create PDF document
    pdf_filename = "Plant_Disease_Report.pdf"
    pdf_path = os.path.join(app.config['DOWNLOAD_FOLDER'], pdf_filename)

    # Create a SimpleDocTemplate object
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()

    # Map languages to their respective fonts
    font_mapping = {
        "en": "NotoSans",
        "hi": "NotoSansHindi",
        "te": "NotoSansTelugu",
        "ml": "NotoSansMalayalam",
        "kn": "NotoSansKannada",
        "pa": "NotoSansGurmukhi",
        "mr": "NotoSansMarathi",
        "bn": "NotoSansBengali",
        "gu": "NotoSansGujarati"
    }

    # Get the font name for the selected language
    font_name = font_mapping.get(language, "NotoSans")

    # Add custom styles for the selected font
    styles.add(ParagraphStyle(name="CustomTitle", fontName=font_name, fontSize=18, leading=22))
    styles.add(ParagraphStyle(name="CustomHeading2", fontName=font_name, fontSize=14, leading=18))
    styles.add(ParagraphStyle(name="CustomNormal", fontName=font_name, fontSize=12, leading=16))
    styles.add(ParagraphStyle(name="Italic", fontName=font_name, fontSize=10, textColor=colors.grey))

    # Create a list to hold the PDF content
    story = []

    # Title
    title = Paragraph("Plant Disease Detection Report", styles["CustomTitle"])
    story.append(title)
    story.append(Spacer(1, 12))

    # Disease Detected
    disease_text = f"<b>Disease Detected:</b> {prediction}"
    disease_para = Paragraph(disease_text, styles["CustomNormal"])
    story.append(disease_para)
    story.append(Spacer(1, 12))

    # Confidence Score
    confidence_text = f"<b>Confidence Score:</b> {round(float(confidence), 2) * 100}%"
    confidence_para = Paragraph(confidence_text, styles["CustomNormal"])
    story.append(confidence_para)
    story.append(Spacer(1, 12))

    # Advice Section
    advice_title = Paragraph("<b>Prevention & Treatment Advice:</b>", styles["CustomHeading2"])
    story.append(advice_title)
    story.append(Spacer(1, 6))

    # Split advice into bullet points
    advice_list = advice.split('. ')
    for tip in advice_list:
        if tip.strip():
            tip_para = Paragraph(f"• {tip.strip()}.", styles["CustomNormal"])
            story.append(tip_para)
            story.append(Spacer(1, 6))

    # Footer with Timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    footer = Paragraph(f"<i>Generated on: {timestamp}</i>", styles["Italic"])
    story.append(Spacer(1, 12))
    story.append(footer)

    # Build the PDF
    doc.build(story)

    return send_file(pdf_path, as_attachment=True)

@app.route('/live')
def live_page():
    return render_template('live.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Preprocess and predict
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = np.expand_dims(img, axis=0).astype(np.float32) / 255.0

        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]['index'])[0]
        class_id = np.argmax(predictions)
        confidence = predictions[class_id]
        label = labels[class_id] if class_id < len(labels) else "Unknown"

        # Draw results on frame
        cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

if __name__ == '__main__':
    app.run(debug=True)
