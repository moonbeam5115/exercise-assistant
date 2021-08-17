import cv2

webcam = cv2.VideoCapture(0)

while webcam.isOpened():
    success, image = webcam.read()
    if not success:
        continue

    cv2.imshow('Testing Camera', image)
        
    if cv2.waitKey(5) & 0xFF == 27:
        break

webcam.release()
cv2.destroyAllWindows()