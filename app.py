from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
import os

app = Flask(__name__)

# Load the trained model
model = load_model('plant_disease_model.h5')

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the uploaded image
    img = request.files['image']

    # Save the image to a temporary file
    img_path = 'temp.jpg'
    img.save(img_path)

    # Preprocess the image
    img = image.load_img(img_path, target_size=(256, 256))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)

    # Make a prediction
    prediction = model.predict(img)

    # Get the class label with the highest probability
    class_label = np.argmax(prediction[0])

    # Convert the class label to a native Python integer
    class_label = int(class_label)

    # Remove the temporary file
    os.remove(img_path)

    # Return the class label as a JSON response
    return jsonify({'class_label': class_label})

@app.route('/result', methods=['GET'])
def result():
    return render_template('result.html')

if __name__ == '__main__':
    app.run(debug=True)