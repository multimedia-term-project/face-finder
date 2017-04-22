import cv2
import numpy

def find_faces():
    image = cv2.imread("hayley-looking-out-window-original.jpg")
    faceCascade = cv2.CascadeClassifier("haarcascade_profileface.xml")
    grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(grayimage)

    for (x, y, w, h) in faces:
        subface = image[y:y+h, x:x+w]
        cv2.imwrite("faces_{}.jpeg".format(y), subface)

def template_matching():
    templates = [cv2.imread("faces_220.jpeg"),
                 cv2.imread("faces_258.jpeg"),
                 cv2.imread("faces_1115.jpeg"),
                 cv2.imread("faces_1152.jpeg")]
    files = [
        "faces_220.jpeg",
        "faces_258.jpeg",
        "faces_1115.jpeg",
        "faces_1152.jpeg"
    ]
    image = cv2.imread("faces_1115.jpeg")
    grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    for i in range(len(templates)):
        res = cv2.matchTemplate(grayimage, cv2.cvtColor(templates[i], cv2.COLOR_BGR2GRAY), cv2.TM_CCOEFF)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        if max_val > 50000000:
            print("Found Face: {}".format(files[i]))



template_matching()

