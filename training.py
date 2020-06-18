import numpy as np
import cv2
import os

import Face_Recognition as fr
print(fr)

test_img=cv2.imread('Resources/sabiya.jpg')
faces_detected,gray_img=fr.faceDetection(test_img)
print("Face Detected: ",faces_detected)

#Training begins
faces,faceID=fr.labels_for_training_data('Resources/images')
face_recognizer=fr.train_classifier(faces,faceID)
face_recognizer.save('Resources/trainingData.yml')

name={0:'Sabiya',1:"Soha"}

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)
    print("Confidence :", confidence)
    print("label :", label)
    fr.draw_rect(test_img,face)
    predict_name=name[label]
    fr.put_text(test_img,predict_name,x,y)

resized_img=cv2.resize(test_img,(1000,700))
cv2.imshow("face detection ", resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
