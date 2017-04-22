import cv2
import numpy
import json
import boto3
import botocore
import io

def find_faces(image):
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

def get_image_from_S3():
    cred = json.load(open("aws.config.json"))
    s3 = boto3.client(
        's3',
        aws_access_key_id=cred["accessKeyId"],
        aws_secret_access_key=cred["secretAccessKey"]
    )
    obj = boto3.resource('s3', aws_access_key_id=cred["accessKeyId"], aws_secret_access_key=cred["secretAccessKey"],  config=botocore.config.Config("us-east-2"))\
        .Bucket('multimedia-term-project')\
        .Object('ryW52UFAe-3-faces.jpg')
    buffer = obj.get()["Body"].read()
    nparr = numpy.fromstring(buffer, numpy.ubyte)
    img_np = cv2.imdecode(nparr, cv2.COLOR_BGR2GRAY)
    find_faces(img_np)

get_image_from_S3()