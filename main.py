import cv2
import numpy
import json
import boto3
import botocore
import os


def get_s3():
    cred = json.load(open("aws.config.json"))
    return boto3.resource('s3', aws_access_key_id=cred["accessKeyId"], aws_secret_access_key=cred["secretAccessKey"],
                        config=botocore.config.Config("us-east-2"))

def find_faces(image):
    faceCascade = cv2.CascadeClassifier("haarcascade_profileface.xml")
    grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(grayimage)

    for (x, y, w, h) in faces:
        subface = image[y:y+h, x:x+w]
        filename = "faces_{}.jpeg".format(y)
        cv2.imwrite(filename, subface)
        im = open(filename, 'rb')
        get_s3().Bucket('multimedia-term-project').Object(filename).put(Body=im, ACL='public-read')
        im.close()
        os.remove(filename)


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

def get_image_from_S3():

    obj = get_s3().Bucket('multimedia-term-project').Object('ryW52UFAe-3-faces.jpg')

    buffer = obj.get()["Body"].read()

    nparr = numpy.fromstring(buffer, numpy.ubyte)
    img_np = cv2.imdecode(nparr, cv2.COLOR_BGR2GRAY)
    
    find_faces(img_np)

get_image_from_S3()