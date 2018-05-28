import cv2

face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def detectFaces(img):
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
    else:
        gray = img
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)
    result = []
    for (x, y, width, height) in faces:
        result.append((x, y, x+width, y+height))
    return result


def drawFaces(img):
    faces = detectFaces(img)
    if faces:
        for (x1, y1, x2, y2) in faces:
            cv2.rectangle(img, (x1, y1), (x2, y2), (225, 105, 65), 3)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img,'FaceDetection', (x1, y1-5), font, 1,(255, 0, 255), 2)
    return img, faces

if __name__ == '__main__' :
    camera = cv2.VideoCapture(0)

    # 遍历视频的每一帧
    while True:
        (grabbed, frame) = camera.read()
        if grabbed:
            img, _ = drawFaces(frame)
            cv2.imshow("ooo", img)
            #press 'q' to quit
            c = cv2.waitKey(1)
            if c&0xFF == ord('q'):
                break
