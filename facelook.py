import cv2  # type: ignore
import mediapipe as mp  # type: ignore
import numpy as np  # type: ignore
import time

class FaceLook:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Cannot open webcam.")
            return
            
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh()
        self.danger_time = 0  # 주위 살피는 시간 누적 변수 추가

    def calculate_rotation_angle(self, left_eye, right_eye, nose_tip):
        left_eye_center = np.mean(left_eye, axis=0)
        right_eye_center = np.mean(right_eye, axis=0)
        eye_center = (left_eye_center + right_eye_center) / 2
        nose_vector = nose_tip - eye_center
        angle = np.arctan2(nose_vector[1], nose_vector[0])
        angle = np.degrees(angle)
        return angle

    def capture(self):
        start_time = time.time()
        danger_start_time = None
        last_angle_log_time = start_time  # 마지막 각도 출력 시간을 저장

        while time.time() - start_time < 12:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            image_height, image_width, _ = frame.shape
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(image_rgb)

            landmarks = []
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for landmark in face_landmarks.landmark:
                        x = int(landmark.x * image_width)
                        y = int(landmark.y * image_height)
                        landmarks.append((x, y))

                nose_tip = np.array(landmarks[1])  # 코 끝점
                left_eye = np.array(landmarks[133:153])  # 왼쪽 눈
                right_eye = np.array(landmarks[362:382])  # 오른쪽 눈

                angle = self.calculate_rotation_angle(left_eye, right_eye, nose_tip)

                # 1초마다 각도를 터미널에 출력
                if time.time() - last_angle_log_time >= 1:
                    print(f"현재 각도: {angle:.2f}도")
                    last_angle_log_time = time.time()

                # 위험한 각도 범위에 해당하는 시간 누적
                if angle <= -160 or angle >= -25:
                    if danger_start_time is None:
                        danger_start_time = time.time()
                    else:
                        # 각도가 위험 범위에 있을 때 경과 시간을 누적
                        self.danger_time += time.time() - danger_start_time
                    danger_start_time = time.time()
                else:
                    danger_start_time = None

                # 얼굴 랜드마크 시각화
                cv2.circle(frame, tuple(nose_tip), 5, (0, 0, 255), -1)
                for landmark in landmarks:
                    cv2.circle(frame, tuple(landmark), 2, (0, 255, 0), -1)

            cv2.imshow('Face Landmarks', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap.release()
        cv2.destroyAllWindows()

        # 최종적으로 주위 살피는 행동이 3초 이상이면 출력
        if self.danger_time >= 3:
            print("주위살피는 행동입니다")
        else:
            print("정상적인 행동입니다")

    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()

# 프로그램 실행
face_look = FaceLook()
face_look.capture()
