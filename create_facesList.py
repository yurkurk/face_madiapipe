import cv2
import face_recognition
import os
import pickle
import time
import csv

path = 'ImagesAttendance'
classNames = []
myList = os.listdir(path)


def find_encodings(images):
    start = time.time()
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encode_list.append(encode)
    end = time.time()
    print('Seconds to encode: ', end-start)
    return encode_list


def create_csv():
    start = time.time()
    with open("list_images.csv", 'w') as f:
        with open('encodings', 'wb') as fp:
            writer = csv.writer(f)
            writer.writerow(['Name', 'Path', 'Encode'])
            encodes = []
            for cls in myList:
                image_path = f'{path}/{cls}'
                currentImg = cv2.imread(image_path)
                classNames.append(os.path.splitext(cls)[0])
                img = cv2.cvtColor(currentImg, cv2.COLOR_BGR2RGB)
                encode = face_recognition.face_encodings(img)[0]
                encodes.append(encode)
                writer.writerow([cls.split()[0], image_path, encode])
            pickle.dump(encodes, fp)
        end = time.time()
        print("Time for encodings creation", end - start)
        print(".csv file created")


if __name__ == '__main__':
    create_csv()
    with open('encodings', 'rb') as fp:
        itemlist = pickle.load(fp)
        print(itemlist)


