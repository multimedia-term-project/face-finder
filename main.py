import cv2
import numpy

def find_faces():
    image = cv2.imread("3-faces.jpg")
    faceCascade = cv2.CascadeClassifier("haarcascade_profileface.xml")
    grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(grayimage)

    for (x, y, w, h) in faces:
        subface = image[y:y+h, x:x+w]
        cv2.imwrite("faces_{}.jpeg".format(y), subface)

    cv2.imshow("Faces found", image)
    cv2.waitKey(0)


find_faces()




