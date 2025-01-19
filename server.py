import os
import sys
import webbrowser
import cv2
import time
import subprocess
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, FileResponse, PlainTextResponse
import tensorflow as tf
from tensorflow.keras.models import load_model # type: ignore
import mediapipe as mp
import numpy as np
from ultralytics import YOLO
from fer import FER

app = FastAPI()



class_names = {0: 'ear', 1: 'eye', 2: 'mouth', 3: 'nose'}
model = YOLO(r'E:\PHICURE\atm\best.pt')

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
emotion_detector = FER()
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()
phone_model = YOLO(r'E:\PHICURE\atm\Phone_Recognition\Phone_Recognition\best.pt', verbose=False)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
#휴대폰 mediapipe 추가코드들
actions = ['call']  # 'call' 동작
seq_length = 30
motion_model = load_model(r'E:\PHICURE\atm\motion_recognition_CALL\models\model2_1.0.h5')
# MediaPipe hands 모델 추가 (이미 위에 정의된 hands와 동일하지만, 변수명을 구분하여 사용)
motion_hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
# MediaPipe 얼굴 감지 모델 초기화
mp_face = mp.solutions.face_detection
face_detection = mp_face.FaceDetection(min_detection_confidence=0.5)
seq=[]




total_count = 0
danger_start_time = None
cap = None

def open_video_capture():
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)

def close_video_capture():
    global cap
    if cap is not None and cap.isOpened():
        cap.release()

def calculate_rotation_angle(left_eye, right_eye, nose_tip):
    left_eye_center = np.mean(left_eye, axis=0)
    right_eye_center = np.mean(right_eye, axis=0)
    eye_center = (left_eye_center + right_eye_center) / 2
    nose_vector = nose_tip - eye_center
    angle = np.arctan2(nose_vector[1], nose_vector[0])
    angle = np.degrees(angle)
    return angle




def show_frame(frame, text=None):
    if text:
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = frame.shape[0] - 20  
        cv2.putText(frame, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)  # 파란색
    cv2.imshow('Detection', frame)




def video_stream(): 
     
    open_video_capture()
    print("프로그램 시작합니다") 
    time.sleep(5)  # 대기 시간
    start_time = time.time()
    face_confirmed = False
    global total_count,cap
    
    while True:
        total_count = 0 
        open_video_capture()
        print("프로그램 시작...")
    

        while cap.isOpened():
          ret, frame = cap.read()
          if not ret:
            break

          frame = cv2.flip(frame, 1) 
          results = model(frame, imgsz=640, conf=0.6)  
          detected_classes = [class_names.get(int(box.cls), 'unknown') for box in results[0].boxes]
          detected_classes_set = set(detected_classes)
          required_classes = {'eye', 'nose', 'mouth'}
          confirmed = required_classes.issubset(detected_classes_set)

        
          gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
          faces = face_cascade.detectMultiScale(gray, 1.1, 4)

          for (x, y, w, h) in faces:
             cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

          if confirmed:
              print("Face Confirmed__눈코입 인식 문제없음")
              face_confirmed = True
              start_time = time.time()  
              while True:
        
                 elapsed_time = time.time() - start_time
                 if elapsed_time > 5:
                   break
               
              cv2.putText(frame, 'Confirmed', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2, cv2.LINE_AA)

              _, jpeg = cv2.imencode('.jpg', frame)
              frame_bytes = jpeg.tobytes()

              yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
           
              break

            
          else:
            if time.time() - start_time >= 18:
                print("눈코입 인식 문제발생. 경고문 띄움")
                cv2.putText(frame, 'Not Detected', (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2, cv2.LINE_AA)
                total_count += 1

                
                if total_count==1:
                    warning_path = os.path.abspath("E:\\PHICURE\\atm\\UIUX\\html\\warning.html")
                    webbrowser.open(f"file://{warning_path}")
                    time.sleep(7)  # 대기 시간

                if total_count == 2:
                   warning_path = os.path.abspath("E:\\PHICURE\\atm\\UIUX\\html\\warning2.html")
                   webbrowser.open(f"file://{warning_path}")
                   time.sleep(10)  # 대기 시간
    
         
                elif total_count == 3:
                   print("눈코입 인식 3번 실패, 본인인증 시작")
                   html_path = os.path.abspath("E:\\PHICURE\\atm\\UIUX\\html\\social_login.html")
                   webbrowser.open(f"file://{html_path}")
                   close_video_capture()
                   cv2.destroyAllWindows()

                   break

          annotated_frame = results[0].plot()
        
          _, jpeg = cv2.imencode('.jpg', annotated_frame)
          frame_bytes = jpeg.tobytes()
          yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

          if cv2.waitKey(1) & 0xFF == ord('q'):
             break

        if face_confirmed:
           total_count = 0 
           print("Starting Emotion Detection")
           start_time = time.time()
           positive_start_time = None
           extract_duration = 16
           overall_emotion = None
           negative_instance_count = 0
           positive_instance_count = 0


           while True:
                ret, input_image = cap.read()
                if not ret:
                  break

                input_image = cv2.flip(input_image, 1)
                gray = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

            
            
                for (x, y, w, h) in faces:
                  face_region = input_image[y:y + h, x:x + w]
                  emotions = emotion_detector.detect_emotions(face_region)
            

                for (x, y, w, h) in faces:
                   face_region = input_image[y:y + h, x:x + w]
                   emotions = emotion_detector.detect_emotions(face_region)


                   if emotions:
                      detected_emotions = emotions[0]["emotions"]
                      max_score = max(detected_emotions.values())
                      max_emotion = max(detected_emotions, key=detected_emotions.get)
                    
                      text_start_x = x + w + 10 
                      text_start_y = max(30, y)  
                   
                      for index, (emotion_name, score) in enumerate(detected_emotions.items()):
                          color = (0, 0, 255) if emotion_name == max_emotion else (255, 0, 0)
                          emotion_score = "{}: {:.2f}".format(emotion_name, score)
                          text_y = text_start_y + index * 20
                          cv2.putText(input_image, emotion_score, (text_start_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

                      if max_emotion in ["angry", "disgust", "fear", "sad", "surprise"]:
                          negative_instance_count += 1
                      else:
                          positive_instance_count += 1
 
                    
                      overall_emotion = "negative" if negative_instance_count > positive_instance_count else "positive"
                      overall_text_y = text_start_y - 30
                      cv2.putText(input_image, f"Overall: {overall_emotion}", (x, overall_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

                _, jpeg = cv2.imencode('.jpg', input_image)
                frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\n'
                     b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')    
            
                if overall_emotion == "positive":
                        if positive_start_time is None:
                            positive_start_time = time.time()  
                        elif time.time() - positive_start_time >= 8:  
                          print("8초 이상 긍정 상태 유지__통과")
                          break
                else:
                        positive_start_time = None  
            


                if time.time() - start_time > extract_duration:
                        if negative_instance_count > positive_instance_count:
                            print("표정인식 문제 발생")
                            total_count += 1
                            warning_path = os.path.abspath("E:\\PHICURE\\atm\\UIUX\\html\\warning3.html")
                            webbrowser.open(f"file://{warning_path}")
                            time.sleep(5)  
                        break

                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            


        print("주변 살피는 코드") 
        open_video_capture()
        danger_time = 0  # danger_time 초기화
        last_angle_log_time = start_time  # 마지막 각도 출력 시간을 저장
        start_time = time.time()

        while time.time() - start_time < 10:
            ret, frame = cap.read()
            if not ret:
                print("프레임을 읽어오지 못했습니다. 카메라 연결을 확인하세요")
                break

            frame = cv2.flip(frame, 1)
            image_height, image_width, _ = frame.shape

            cv2.putText(frame, 'Surrounding Behavior Analysis', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(image_rgb)
            landmarks = []

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    for landmark in face_landmarks.landmark:
                        x = int(landmark.x * image_width)
                        y = int(landmark.y * image_height)
                        landmarks.append((x, y))

                    # 눈, 코, 입 위치를 기반으로 고개 움직임 분석
                    nose_tip = np.array(landmarks[1])
                    left_eye = np.array(landmarks[133:153])
                    right_eye = np.array(landmarks[362:382])
                    angle = calculate_rotation_angle(left_eye, right_eye, nose_tip)

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
                            danger_time += time.time() - danger_start_time
                        danger_start_time = time.time()
                    else:
                        danger_start_time = None

                    cv2.circle(frame, tuple(nose_tip), 5, (0, 0, 255), -1)
                    for landmark in landmarks:
                        cv2.circle(frame, tuple(landmark), 2, (0, 255, 0), -1)


            # 프레임을 JPEG로 인코딩하여 클라이언트에게 전송
            _, jpeg = cv2.imencode('.jpg', frame)
            frame_bytes = jpeg.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


        if danger_time >= 3:
            print("주위살피는 행동입니다")
            warning_path = os.path.abspath("E:\\PHICURE\\atm\\UIUX\\html\\warning_look.html")
            webbrowser.open(f"file://{warning_path}")
            time.sleep(5)  # 대기 시간
        else:
            print("주위살피는 행동 감지 안됨 ㅡ 통과")

                
            

        print("휴대폰 객체 감지")
        print(f"TensorFlow version in hand_detection_analysis.py: {tf.__version__}")
        phone_model = YOLO(r'E:\PHICURE\atm\Phone_Recognition\Phone_Recognition\best.pt', verbose=False) 
        start_time = time.time()
        phone_detected = False


        with mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
            while time.time() - start_time < 11:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Frame could not be read.")
                    break
        
                frame = cv2.flip(frame, 1)
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
                results = phone_model(frame)
                for result in results:
                    boxes = result.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        confidence = box.conf[0]
                        label = box.cls[0]
                        if label == 0: 
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f'Phone {confidence:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                            phone_detected = True

    
                hand_results = hands.process(img_rgb)
                if hand_results.multi_hand_landmarks:
                    for hand_landmarks in hand_results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            mp_hands.HAND_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
                        )

                

            
                _, jpeg = cv2.imencode('.jpg', frame)
                frame_bytes = jpeg.tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        if phone_detected == True:
            print("휴대폰 객체 감지")
            warning_path = os.path.abspath("E:\\PHICURE\\atm\\UIUX\\html\\warning4.html")
            webbrowser.open(f"file://{warning_path}")
            total_count+=1
            time.sleep(5)
        
            
                

        
  

    
        
        if total_count >= 2:
          print("두 개 조건 이상 감지, 본인인증 시작")
          html_path = os.path.abspath("E:\\PHICURE\\atm\\UIUX\\html\\social_login.html")
          webbrowser.open(f"file://{html_path}")
          total_count = 0
          time.sleep(7)  # 대기 시간

          


        else:
          print("두 개 조건 이상 감지되지 않음")
          total_count = 0
          
        
        if cv2.waitKey(1) & 0xFF == ord('q'):  
            print("프로그램 종료 요청 ")
            break 

        
    close_video_capture()
    cv2.destroyAllWindows()
    print("비디오 스트림 종료")



if __name__ == "__main__":
    video_stream()

         



   


@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(video_stream(), media_type='multipart/x-mixed-replace; boundary=frame')


@app.get("/")
async def read_root():
    return FileResponse("E:\\PHICURE\\atm\\UIUX\\html\\videoCam.html")

@app.get("/uiux")
async def read_uiux():
    return FileResponse("E:\\PHICURE\\atm\\UIUX\\html\\uiux.html")

@app.get("/uiux2")
async def read_uiux2():
    return FileResponse("E:\\PHICURE\\atm\\UIUX\\html\\uiux2.html")

@app.get("/send")
async def read_send():
    return FileResponse("E:\\PHICURE\\atm\\UIUX\\html\\send.html")

@app.get("/tongjang")
async def read_tongjang():
    return FileResponse("E:\\PHICURE\\atm\\UIUX\\html\\tongjang.html")

@app.get("/withdraw")
async def read_withdraw():
    return FileResponse("E:\\PHICURE\\atm\\UIUX\\html\\withdraw.html")

@app.get("/card")
async def read_card():
    return FileResponse("E:\\PHICURE\\atm\\UIUX\\html\\card.html")

@app.get("/fin")
async def read_fin():
    return FileResponse("E:\\PHICURE\\atm\\UIUX\\html\\fin.html")

@app.get("/start_uiux")
async def read_start_uiux():
    return FileResponse("E:\\PHICURE\\atm\\UIUX\\html\\start_uiux.html")

@app.get("/account_send")
async def read_account_send():
    return FileResponse("E:\\PHICURE\\atm\\UIUX\\html\\account_send.html")

@app.get("/account_send2")
async def read_account_send2():
    return FileResponse("E:\\PHICURE\\atm\\UIUX\\html\\account_send2.html")




# 여러 HTML 파일 서빙 엔드포인트
@app.get("/{file_name}")
async def serve_html(file_name: str):
    base_path = "E:\\PHICURE\\atm\\UIUX\\html\\"
    file_path = os.path.join(base_path, file_name)
    if os.path.exists(file_path):
        return FileResponse(file_path)
    else:
        return PlainTextResponse(f"{file_name} not found", status_code=404)
    




