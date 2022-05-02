import cv2
import numpy as np
import face_recognition
import time
from datetime import datetime
import pandas as pd
from face_depth import measure_depth, putTextRect
from head_pose import head_pose_processing
from cvzone.FaceMeshModule import FaceMeshDetector
import pickle


list_images = pd.read_csv('list_images.csv')
path = 'ImagesAttendance'
classNames = list_images['Name'].values
paths = list_images['Path'].values
with open('encodings', 'rb') as fp:
    encodeLstKnown = pickle.load(fp)
confidence = 0.65


def mark_attendance(name):
    with open('attendance.csv', 'r+') as f:
        my_data_list = f.readlines()
        nameList = []
        for line in my_data_list:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dt_string = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name}, {dt_string}')


def start_stream(cap, detector):
    while True:
        start = time.time()
        success, img = cap.read()
        d = measure_depth(img, detector, draw=False)
        img = head_pose_processing(img)
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        facesCurFrame = face_recognition.face_locations(imgS, model='hog')
        encodingsCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodingsCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeLstKnown, encodeFace)
            face_dist = face_recognition.face_distance(encodeLstKnown, encodeFace)
            match_index = np.argmin(face_dist)
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            if d < 75:
                if face_dist[match_index] < confidence:
                    if matches[match_index]:
                        name = classNames[match_index].upper()
                        cv2.putText(img, name, (x1+6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                        mark_attendance(name)
                else:
                    cv2.putText(img, "Unknown", (x1+6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            else:
                cv2.putText(img, "Too far from webcam", (x1, y2), cv2.FONT_HERSHEY_COMPLEX,
                            1, (0, 255, 255), 2)
        end = time.time()
        totalTime = end - start
        fps = 1 / totalTime
        cv2.putText(img, f'Face Distance: {int(d)}', (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.putText(img, f'FPS: {int(fps)}', (20, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
        cv2.imshow("Webcam", img)

        cv2.waitKey(1)


if __name__ == "__main__":
    detector = FaceMeshDetector(maxFaces=1)
    with open('attendance.csv', 'w') as fp:
        pass
    cap = cv2.VideoCapture(0)
    start_stream(cap, detector)

