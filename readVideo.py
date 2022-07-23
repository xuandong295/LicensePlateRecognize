import cv2
import uuid
cap = cv2.VideoCapture('license_video.mp4')
# if (cap.isOpened()== False):
#     print("Error opening video stream or file")
while True:
# Capture frame-by-frame
    ret, frame = cap.read()

    # Display the resulting frame

    cv2.imshow('Frame', frame)
    cv2.imwrite("MyFile/%s.jpg" % uuid.uuid4(), frame)
    # Press Q on keyboard to  exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
# Closes all the frames
cv2.destroyAllWindows()


# import cv2
# #lấy hình ảnh từ camera đầu tiên
# cap = cv2.VideoCapture(0 , cv2.CAP_DSHOW)
# while(cap.isOpened()):
#     #Đọc video
#     ret, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     #Hiển thị video đen trắng
#     cv2.imshow('frame',gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cap.release()
# cv2.destroyAllWindows()