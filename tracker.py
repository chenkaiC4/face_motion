
'''
https://www.learnopencv.com/object-tracking-using-opencv-cpp-python/


'''

import cv2
import sys

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

 
(major_ver, minor_ver, subminor_ver) = (cv2.__version__).split(".");

if __name__ == '__main__' :
 
    # Set up tracker.
    # Instead of MIL, you can also use
 
    tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
    tracker_type = tracker_types[2]
 
    if int(minor_ver) < 3:
        tracker = cv2.Tracker_create(tracker_type)
    else:
        if tracker_type == 'BOOSTING':
            tracker = cv2.TrackerBoosting_create()
        if tracker_type == 'MIL':
            tracker = cv2.TrackerMIL_create()
        if tracker_type == 'KCF':
            tracker = cv2.TrackerKCF_create()
        if tracker_type == 'TLD':
            tracker = cv2.TrackerTLD_create()
        if tracker_type == 'MEDIANFLOW':
            tracker = cv2.TrackerMedianFlow_create()
        if tracker_type == 'GOTURN':
            tracker = cv2.TrackerGOTURN_create()
 
    # Read video
    video = cv2.VideoCapture(0)
        
    # Exit if video not opened.
    if not video.isOpened():
        print("Could not open video")
        sys.exit()

    ok, first_frame = video.read()
    if not ok:
        print('Cannot read video file')
        sys.exit()

    multi_tracker = cv2.MultiTracker_create()
    detectingFace = True

    trackers = []
    while True:
        # Read a new frame
        ok, frame = video.read()
        if not ok:
            print('Cannot read video file')
            sys.exit()

        # 检测面部
        if detectingFace:
            faces = detectFaces(frame)
            if len(faces) != 0:
                # init trackers
                print("===> detacted face")
                detectingFace = False
                for facebbox in faces:
                    tracker = cv2.TrackerKCF_create()
                    ok = tracker.init(frame, facebbox)
                    # ok = multi_tracker.add(cv2.TrackerKCF_create(), frame, facebbox)
                    trackers.append(tracker)
                    if not ok:
                        print("初始化 tracker 失败")
                        sys.exit()
            continue
        
        print("tracking...")
        # Start timer
        timer = cv2.getTickCount()
 
        # Update tracker
        bboxs = []
        for tracker in trackers:
            ok, bbox = tracker.update(frame)
            if ok:
                bboxs.append(bbox)
        # ok, bboxs = multi_tracker.update(frame)
 
        # Draw bounding box
        if ok:
            # Tracking success
            for bbox in bboxs:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[2]), int(bbox[3]))
                cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        else :
            detectingFace = True
            print("===> tracker failed")
            # Tracking failure
            cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)
            continue

        # Display tracker type on frame
        cv2.putText(frame, tracker_type + " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2)
     
        # Calculate Frames per second (FPS)
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        # Display FPS on frame
        cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
 
        # Display result
        cv2.imshow("Tracking", frame)
 
        # Exit if ESC pressed
        k = cv2.waitKey(1) & 0xff
        if k == 27 : break
