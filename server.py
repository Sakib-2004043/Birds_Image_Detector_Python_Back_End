from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
from PIL import Image
import io
import json
import os

# App Instance
app = Flask(__name__)
CORS(app)

# Load the model and labels once at startup
try:
    model = tf.keras.models.load_model('Model/BirdsModel.keras')
except Exception as e:
    model = None
    print(f"Error loading model: {str(e)}")

try:
    with open("Data/labels.json", 'r') as file:
        labels = json.load(file)
    labels = dict((v, k) for k, v in labels.items())  # Reverse for index-to-label mapping
except Exception as e:
    labels = None
    print(f"Error loading labels: {str(e)}")

@app.route("/api/upload", methods=['POST'])
def upload_image():
    if 'image' not in request.files:
        return jsonify({"message": "No image part in the request"}), 400

    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({"message": "No selected file"}), 400

    if model is None or labels is None:
        return jsonify({"message": "Model or labels not loaded properly"}), 500

    try:
        # Open and convert the image
        img = Image.open(image_file.stream)
        # img = keras_image.load_img("Image/crow.png", target_size=(64, 64)) 

        # Save the image to a folder
        # save_folder = 'Upload'
        # img_path = os.path.join(save_folder, image_file.filename)
        # img.save(img_path)

        img_resized = img.resize((64, 64))
        # Preprocess the image
        test_image = keras_image.img_to_array(img_resized)
        test_image = np.expand_dims(test_image, axis=0)
        test_image = test_image / 255.0  # Normalize the image to [0, 1] range

        # Predict the class
        result = model.predict(test_image)
        index = np.argmax(result, axis=1)[0]
        prediction = labels.get(index, "Unknown")

        return jsonify({"message": "Image uploaded and detected successfully", "prediction": prediction}), 200

    except Exception as e:
        return jsonify({"message": f"Failed to process image: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8080)
