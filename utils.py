from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
import os

app = Flask(__name__)
UPLOAD_FOLDER = "static/uploads"
RESULT_FOLDER = "static/results"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

# Load pre-trained model
weights_path = "models/colorization_release_v2.caffemodel"
config_path = "models/colorization_deploy_v2.prototxt"
pts_path = "models/pts_in_hull.npy"

net = cv2.dnn.readNetFromCaffe(config_path, weights_path)
pts = np.load(pts_path).transpose().reshape(2, 313, 1, 1).astype("float32")
class8 = net.getLayerId("class8_ab")
conv8 = net.getLayerId("conv8_313_rh")
net.getLayer(class8).blobs = [pts]
net.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype="float32")]

def l_channel_from_lab_image(lab):
    """Get the lightness channel (L) from a Lab image"""
    return cv2.split(lab)[0]

def rgb_from_l_and_ab(L, ab):
    """Combine L channel & ab channel to RGB image"""
    lab = np.concatenate((L[:, :, np.newaxis], ab), axis=2)
    rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    rgb = np.clip(rgb, 0, 1)
    rgb = (255 * rgb).astype("uint8")
    return rgb

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'image' not in request.files:
        return "No file part"
    file = request.files['image']
    if file.filename == '':
        return "No selected file"
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)
    
    # Process image
    image = cv2.imread(filepath)
    height, width = image.shape[:2]
    image = image.astype("float32") / 255.0
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    # Extract L channel and process with the model
    L = l_channel_from_lab_image(lab)
    L_resized = cv2.resize(L, (224, 224)) - 50
    net.setInput(cv2.dnn.blobFromImage(L_resized))
    ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
    ab = cv2.resize(ab, (width, height))
    
    # Combine L and ab channels
    colorized = rgb_from_l_and_ab(l_channel_from_lab_image(lab), ab)
    
    result_path = os.path.join(RESULT_FOLDER, "colorized.png")
    cv2.imwrite(result_path, colorized)
    
    return result_path

@app.route('/download')
def download():
    return send_file("static/results/colorized.png", as_attachment=True)

if __name__ == '__main__':
    app.run(debug=True)