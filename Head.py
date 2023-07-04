import cv2
import math
from ultralytics import YOLO
import numpy as np

model = YOLO('yolov8n.pt')  # initialize
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

count = 0
center_points_prev_frames = []
frame_count = -1
arr = []

while True:
    ret, frame = cap.read()
    frame_count += 1
    if not ret:
        break

    center_points = []
    
    results = model(frame, show=False, stream=True)

    for r in results:
        arr.append(r)
        boxes = r.boxes
        for box in boxes:
            if int(box.cls) == 0:
                if box.conf < 0.5:
                    continue
                x1, y1, x2, y2 = box.xyxy[0]
                x1 , y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                cx = int((x1 + x2)/2)
                cy = int((y1 + y2)/2)
                center_points.append((cx, cy))

                cv2.rectangle(frame, (x1, y1), (x2, y2), (150, 255, 150), 2)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                cv2.putText(frame, f'{model.names[int(box.cls)]} {float(box.conf):.2f}', 
                            (x1, y1-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
    if center_points_prev_frames == []:
        count = max(count, len(center_points))

    else:
        for i in center_points:
            unique = True
            for k in center_points_prev_frames:
                for j in k:
                    dist = math.hypot(j[0] - i[0], j[1] - i[1])
                    if dist < 60:
                        # print(i,j)
                        unique = False
                        break

                if unique == False:
                    break

            if unique == True:
                count += 1

    cnt = len(center_points)

    cv2. putText(frame, str(count), (10, 50), 0, 1, (255, 255, 255), 2)
    cv2.putText(frame, str(cnt), (10, 90), 0, 1, (255, 255, 255), 2)
    cv2.imshow("frame", frame)

    if frame_count < 100:
        center_points_prev_frames.append(center_points)
    else:
        center_points_prev_frames[frame_count%100] = center_points

    cv2.imshow('frame', frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()

print(arr)