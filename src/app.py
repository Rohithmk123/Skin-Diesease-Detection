# src/app.py
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import gradio as gr

Labels = ['Benign', 'Malignant']
IMAGE_SIZE = (224, 224)
FV_SIZE = 1280

MODULE_HANDLE = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/2"

feature_extractor = hub.KerasLayer(MODULE_HANDLE, input_shape=IMAGE_SIZE + (3,), output_shape=[FV_SIZE])
feature_extractor.trainable = False

# Dummy model for demo (replace with trained weights/load_model for real use)
model = tf.keras.Sequential([
    feature_extractor,
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dropout(rate=0.2),
    tf.keras.layers.Dense(len(Labels), activation='softmax',
                          kernel_regularizer=tf.keras.regularizers.l2(0.0001)),
])

def predict_image(img):
    img = np.array(img) / 255.0
    img_4d = img.reshape(-1, 224, 224, 3)
    prediction = model.predict(img_4d)[0]
    return {Labels[i]: float(prediction[i]) for i in range(len(Labels))}

gr.Interface(
    fn=predict_image,
    inputs=gr.inputs.Image(shape=(224,224)),
    outputs=gr.outputs.Label(num_top_classes=len(Labels)),
    interpretation='default'
).launch()