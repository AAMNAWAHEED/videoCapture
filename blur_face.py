import cv2
cap = cv2.VideoCapture(0)
#model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, img = cap.read()
    cv2.imshow("real image",img)

    face  = model.detectMultiScale(img)
    if len(face) >0:
            x1 = face[0][0]
            y1 = face[0][1]
            x2 = face[0][2] + x1
            y2 = face[0][3] + y1 
            cimg = img[y1:y2 , x1:x2]
            blur_img = cv2.blur(cimg, (50,50))
            img[y1:y2 , x1:x2] = blur_img
            cv2.imshow("blured image",img)
            if cv2.waitKey(100) == 13:
                break

cv2.destroyAllWindows()