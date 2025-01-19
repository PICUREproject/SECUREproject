import cv2

# 웹캠 열기
cap = cv2.VideoCapture(0)

# 프레임 속도 확인
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"웹캠 프레임 속도: {fps} fps")

cap.release()