from flask import Flask, render_template, request, jsonify
import base64
import os
import time

app = Flask(__name__)

# Create folder for Dataset
DATASET_DIR = "dataset"
os.makedirs(DATASET_DIR, exist_ok=True)

# First page for UI to capture image and label
@app.route('/')
def index():
    return render_template('index.html')

# Endpoint for receiving image and label data from the web page
@app.route('/save-sample', methods=['POST'])
def save_sample():
    data = request.json
    label = data.get('label')
    image_data = data.get('image')

    if not label or not image_data:
        return jsonify({"success": False, "error": "ข้อมูลไม่ครบถ้วน"}), 400

    # Create subfolder based on Label (e.g., dataset/46/)
    label_dir = os.path.join(DATASET_DIR, label)
    os.makedirs(label_dir, exist_ok=True)

    # Split the Base64 header (data:image / png;base64,...)
    encoded_data = image_data.split(',')[1]
    
    # Create filename using Label and Timestamp to avoid duplicates
    filename = f"{label}_{int(time.time() * 1000)}.png"
    filepath = os.path.join(label_dir, filename)

    # Convert Base64 to image and save
    with open(filepath, "wb") as fh:
        fh.write(base64.b64decode(encoded_data))

    return jsonify({"success": True, "filename": filename})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
