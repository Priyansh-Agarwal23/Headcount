import cv2
import math
import numpy as np
from object_detection import ObjectDetection

od = ObjectDetection()

cap = cv2.VideoCapture(0)

count = 0
center_points_prev_10_frames = []
frame_count = -1

while True:
    check, img = cap.read()
    frame_count += 1
    if not check:
        break

    center_points = []
    (class_ids, scores, boxes) = od.detect(img)

    for i in range(len(boxes)):
        if class_ids[i] == 0:
            (x,y,w,h) = boxes[i]
            cx = int((x + x + w)/2)
            cy = int((y + y + h)/2)
            center_points.append((cx, cy))

            cv2.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), 2)
            cv2.circle(img, (cx, cy), 5, (0, 0, 255), -1)

    if center_points_prev_10_frames == []:
        count = max(count, len(center_points))
        # print("Count = ", count)

    else:
        for i in center_points:
            unique = True
            for k in center_points_prev_10_frames:
                for j in k:
                    dist = math.hypot(j[0] - i[0], j[1] - i[1])
                    if dist < 40:
                        # print(i,j)
                        unique = False
                        break

                if unique == False:
                    break

            if unique == True:
                count += 1
                print(i)

    cnt = len(center_points)
    count = max(count, cnt)
                
    # print(count)
    cv2. putText(img, str(count), (10, 50), 0, 1, (255, 255, 255), 2)
    cv2.putText(img, str(cnt), (10, 90), 0, 1, (255, 255, 255), 2)
    cv2.imshow("frame", img)

    if frame_count < 10:
        center_points_prev_10_frames.append(center_points)
    else:
        center_points_prev_10_frames[frame_count%10] = center_points

    print(center_points_prev_10_frames)

    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()