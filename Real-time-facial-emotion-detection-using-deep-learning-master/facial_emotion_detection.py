import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from fer import FER
import tensorflow as tf

# 얼굴 감지 모델 로드
face = cv2.CascadeClassifier('haar cascade files/haarcascade_frontalface_alt.xml')

cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL

# Pre-trained FER 모델 사용
emotion_detector = FER()
# 또는, 본인 훈련 모델 사용
# emotion_detector = tf.keras.models.load_model('models/model.h5')

# 부정적인 감정 목록
negative_emotions = ['angry', 'sad', 'fear']

# 정확도 기록용 변수
true_emotions = []
predicted_emotions = []

def update_plot(frame):
    plt.clf()
    if len(true_emotions) > 0:
        accuracy = np.mean(np.array(true_emotions) == np.array(predicted_emotions))
        plt.plot(true_emotions, 'b', label='True Emotions')
        plt.plot(predicted_emotions, 'r', label='Predicted Emotions')
        plt.title(f'Accuracy: {accuracy:.2f}')
        plt.legend()

# 그래프 설정
fig, ax = plt.subplots()
ani = FuncAnimation(fig, update_plot, interval=1000)

while True:
    ret, input_image = cap.read()

    result = emotion_detector.detect_emotions(input_image)
    if result:
        bounding_box = result[0]["box"]
        emotions = result[0]["emotions"]
        cv2.rectangle(input_image, (
            bounding_box[0], bounding_box[1]), (
            bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]),
            (100, 155, 255), 2,)

        # 가장 높은 확률 감정 찾기
        score_max = 0
        dominant_emotion = ""

        for emotion, score in emotions.items():
            if score > score_max:
                score_max = score
                dominant_emotion = emotion

        # 감정 정보 화면에 표시
        for index, (emotion_name, score) in enumerate(emotions.items()):
            color = (0, 0, 255) if emotion_name == dominant_emotion else (255, 0, 0)
            emotion_score = "{}: {:.2f}".format(emotion_name, score)

            cv2.putText(input_image, emotion_score,
                        (20, 20 + index * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        # 감정 예측을 화면에 표시
        cv2.putText(input_image, f"Predicted: {dominant_emotion}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 부정적인 감정만 기록
        if dominant_emotion in negative_emotions:
            true_emotions.append(dominant_emotion)
            predicted_emotions.append(dominant_emotion)

    cv2.imshow('Emotion Detection', input_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.show()
