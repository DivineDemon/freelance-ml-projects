import time
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from PIL import Image
from flask import Flask, render_template, Response

app = Flask(__name__)

cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)

def gen_frames():
    img_count = 0
    while (True) and (img_count != 20):
        success, frame = cam.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode(".jpg", frame)
            cv2.imwrite("image_" + str(img_count) + ".jpg", frame)
            frame = buffer.tobytes()
            yield (b"--frame\r\n" b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
        img_count += 1
        time.sleep(12)


@app.route("/prediction")
def prediction():
    for i in range(0, 20):
        img = Image.open("image_" + str(i) + ".jpg").convert("L").resize((48, 48))
        img = np.asarray(img)
        img = img.reshape(1, 48, 48, 1)
        model = tf.keras.models.load_model("model")
        pred = model.predict(img)
        prediction_file = open("prediction_file.txt", "a")
        prediction_file.write("\n")
        prediction_file.write(str(pred))
    return "Complete"

@app.route("/video_feed")
def video_feed():
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route("/", methods=["GET", "POST"])
def main():
    return render_template("index.html")