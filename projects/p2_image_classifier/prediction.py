import argparse
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from PIL import Image

def load_model(model_path):
    return keras.models.load_model(model_path, custom_objects={'KerasLayer': hub.KerasLayer})

def process_image(image_path):
    image = Image.open(image_path)
    image = np.asarray(image)
    image = tf.image.resize(image, (224, 224)) / 255.0
    return np.expand_dims(image, axis=0)

def predict(image_path, model, top_k):
    image = process_image(image_path)
    predictions = model.predict(image)
    probs = np.sort(predictions[0])[-top_k:][::-1]
    classes = np.argsort(predictions[0])[-top_k:][::-1]
    return probs, classes

def main():
    parser = argparse.ArgumentParser(description='Predict flower name from an image using a trained model.')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('model_path', type=str, help='Path to the saved Keras model')
    parser.add_argument('--top_k', type=int, default=5, help='Return the top K most likely classes')
    parser.add_argument('--category_names', type=str, help='Path to a JSON file mapping labels to flower names')

    args = parser.parse_args()

    model = load_model(args.model_path)
    probs, classes = predict(args.image_path, model, args.top_k)

    if args.category_names:
        with open(args.category_names, 'r') as f:
            class_names = json.load(f)
        flower_names = [class_names[str(cls)] for cls in classes]
    else:
        flower_names = classes

    for prob, flower in zip(probs, flower_names):
        print(f"{flower}: {prob:.4f}")

if __name__ == '__main__':
    main()