import cv2
import mediapipe as mp
import pyautogui
import numpy as np

cam = cv2.VideoCapture(0)
if not cam.isOpened():
    print("Error: Camera not accessible.")
    exit()

face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True, max_num_faces=1)

screen_w, screen_h = pyautogui.size()

smoothening = 0.2
prev_x, prev_y = 0, 0

blink_threshold = 0.02

cv2.namedWindow('Eye Controlled Mouse', cv2.WINDOW_NORMAL)
cv2.setWindowProperty('Eye Controlled Mouse', cv2.WND_PROP_TOPMOST, 1)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    frame_h, frame_w, _ = frame.shape

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0].landmark

        eye_landmark = landmarks[475]
        x = int(eye_landmark.x * frame_w)
        y = int(eye_landmark.y * frame_h)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        screen_x = screen_w * eye_landmark.x
        screen_y = screen_h * eye_landmark.y

        curr_x = prev_x + (screen_x - prev_x) * smoothening
        curr_y = prev_y + (screen_y - prev_y) * smoothening
        pyautogui.moveTo(curr_x, curr_y)
        prev_x, prev_y = curr_x, curr_y

    
        left_eye_top = landmarks[145]
        left_eye_bottom = landmarks[159]
        vert_dist = abs(left_eye_top.y - left_eye_bottom.y)

        for point in [left_eye_top, left_eye_bottom]:
            cx, cy = int(point.x * frame_w), int(point.y * frame_h)
            cv2.circle(frame, (cx, cy), 3, (255, 255, 0), -1)

        if vert_dist < blink_threshold:
            pyautogui.click()
            cv2.putText(frame, "CLICK", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            pyautogui.sleep(0.5)  

    cv2.imshow('Eye Controlled Mouse', frame)

   
    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
