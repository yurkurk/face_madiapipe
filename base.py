import cv2
import numpy as np
import face_recognition

imgIlon = face_recognition.load_image_file("BasicImage/ilon_1.jpg")
imgIlon = cv2.cvtColor(imgIlon, cv2.COLOR_BGR2RGB)
imgTest = face_recognition.load_image_file("BasicImage/ilon_3.jpeg")
imgTest = cv2.cvtColor(imgTest, cv2.COLOR_BGR2RGB)

faceLoc = face_recognition.face_locations(imgIlon)[0]
encodeIlon = face_recognition.face_encodings(imgIlon)[0]
cv2.rectangle(imgIlon, (faceLoc[1], faceLoc[2]),
              (faceLoc[3], faceLoc[0]), (255, 0, 255), 2)

faceLocTest = face_recognition.face_locations(imgTest)[0]
encodeTest = face_recognition.face_encodings(imgTest)[0]
cv2.rectangle(imgTest, (faceLocTest[1], faceLocTest[2]),
              (faceLocTest[3], faceLocTest[0]), (255, 0, 255), 2)

results = face_recognition.compare_faces([encodeIlon], encodeTest)
faceDist = face_recognition.face_distance([encodeIlon], encodeTest)
print(results, faceDist)
cv2.putText(imgTest, f'{results} {round(faceDist[0], 2)}', (50, 50),
            cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

cv2.imshow("Elon Mask", imgIlon)
cv2.imshow("Elon Test", imgTest)
cv2.waitKey(0)