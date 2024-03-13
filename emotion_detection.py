import cv2
import numpy as np
from tensorflow.keras.models import load_model
loaded_model = load_model('saved_model.h5')

cam = cv2.VideoCapture(0)
model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

labels = {0:'Anger',
 1:'Disgust',
 2:'Fear',
 3:'Happiness',
 4:'Sadness',
 5:'Surprise'}


while True:
    ret,frame = cam.read()
    #cv2.imshow("frame",frame)
    face = model.detectMultiScale(frame)

    if len(face) >0:
        for (x,y,w,h) in face:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(255, 0, 0),2)
            crop_img = frame[y:h+y,x:x+w]
            curr_frame = cv2.resize(crop_img,(128,128))
            curr = curr_frame.reshape((1,128,128,3))
            pred = loaded_model.predict(curr)
            emotion=labels[np.argmax(pred)]

            cv2.putText(frame,emotion,(x+20,y-60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255))
        cv2.imshow("image",frame)
        if cv2.waitKey(100) == 13:
            break

cv2.destroyAllWindows()



