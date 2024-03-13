import cv2
cap = cv2.VideoCapture(0)
model  = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

img_counter = 0
while True:
    ret,frame = cap.read()
    cv2.imshow('frame',frame)
    k = cv2.waitKey(1)
    if k%256 == 27:
        print("escape pressed to close the app")
        break
    elif k%256 == 32: #space
        img_name = "videocap_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name,frame)
        print("screenshow take")
        img_counter+=1

cap.release()
cv2.destroyAllWindows()