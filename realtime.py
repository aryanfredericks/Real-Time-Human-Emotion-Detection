import cv2
from tensorflow.keras.models import load_model
import numpy as np
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)
model = load_model('FACE_REG_MODEL.keras')


def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 3)
    return feature/255.0


webcam = cv2.VideoCapture(0)

labels = {0: 'angry', 1: 'disgust', 2: 'fear',
          3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

while True:
    i, im = webcam.read()
    gray = im.copy()
    faces = face_cascade.detectMultiScale(im, 1.3, 5)
    try:
        for (p, q, r, s) in faces:
            image = gray[q:q+s, p:p+r]
            cv2.rectangle(im, (p, q), (p+r, q+s), (255, 0, 0), 2)
            image = cv2.resize(image, (48, 48))
            img = extract_features(image)
            pred = model.predict(img, verbose=0)
            pred = np.argmax(pred, axis=1)
            label = labels[pred[0]]
            cv2.putText(im, label, (p-10, q-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255))
        cv2.imshow("Output", im)
        cv2.waitKey(27)
    except cv2.error:
        pass
