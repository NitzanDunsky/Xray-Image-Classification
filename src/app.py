from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import webbrowser
import os
from inferece import load_and_infer
#initialize app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

# Route for handling the uploaded file and inference
@app.route('/upload', methods=['POST'])

def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file:
        # Save the uploaded file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Run inference on the uploaded file
        result = load_and_infer(file_path)

        # Send back the image path and prediction result to the result page
        return render_template('result.html', image_path=file_path, prediction=result)

@app.route('/uploads/<filename>')
def show_image(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# Automatically open browser when Flask starts
def open_browser():
    webbrowser.open_new("http://localhost:5000/")