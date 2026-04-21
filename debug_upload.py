import cv2

cap = cv2.VideoCapture(0)

# FIX: cascade 파일 경로 (같은 폴더에 있어야 함)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret, frame = cap.read()

    # FIX: ret 체크
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # BUG 3: 일부러 남김 (faces → face 오타)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow('frame', frame)

    # FIX: ESC 키로 종료 (27)
    if cv2.waitKey(1) & 0xFF == 27:
        break

# FIX: 자원 해제
cap.release()
cv2.destroyAllWindows()