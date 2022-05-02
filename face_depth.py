import cv2
from cvzone.FaceMeshModule import FaceMeshDetector


def putTextRect(img, text, pos, scale=3, thickness=3, colorT=(0, 0, 0),
                colorR=(255, 255, 255), font=cv2.FONT_HERSHEY_PLAIN,
                offset=10, border=None, colorB=(0, 255, 0)):
    ox, oy = pos
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)

    x1, y1, x2, y2 = ox - offset, oy + offset, ox + w + offset, oy - h - offset

    # cv2.rectangle(img, (x1, y1), (x2, y2), colorR, cv2.FILLED)
    if border is not None:
        cv2.rectangle(img, (x1, y1), (x2, y2), colorB, border)
    cv2.putText(img, text, (ox, oy), font, scale, colorT, thickness)

    return img, [x1, y2, x2, y1]


def measure_depth(image, det, draw=True):
    img, faces = det.findFaceMesh(image, draw=draw)
    d = 0
    if faces:
        face = faces[0]
        pointLeft = face[145]
        pointRight = face[374]
        w, _ = det.findDistance(pointLeft, pointRight)
        W = 6.3
        # Finding focal length
        # d = 50
        # f = (w * d)/ W
        # Finding distance
        f = 675
        d = (W * f) / w
    return d


if __name__ == "__main__":

    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector(maxFaces=1)

    while True:
        success, img = cap.read()
        print(measure_depth(img, detector, draw=False))

        cv2.imshow("Image", img)
        cv2.waitKey(1)
