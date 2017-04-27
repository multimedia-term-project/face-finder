import json
import os
import time

import cv2
import numpy
import boto3
import botocore
import pika


def get_s3():
    cred = json.load(open("aws.config.json"))
    return boto3.resource('s3', aws_access_key_id=cred["accessKeyId"], aws_secret_access_key=cred["secretAccessKey"],
                        config=botocore.config.Config("us-east-2"))

def find_faces(image, imageName):
    faceCascade = cv2.CascadeClassifier("haarcascade_profileface.xml")
    grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(grayimage)

    for (x, y, w, h) in faces:
        subface = image[y:y+h, x:x+w]
        filename = "{}_faces_{}.jpeg".format(imageName, y)
        cv2.imwrite(filename, subface)
        im = open(filename, 'rb')
        get_s3().Bucket('multimedia-term-project').Object(filename).put(Body=im, ACL='public-read')
        im.close()
        os.remove(filename)

# This function is going to need to be majorly revamped to find faces
# def template_matching():
#     image = cv2.imread("faces_1115.jpeg")
#     grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#
#     for i in range(len(templates)):
#         res = cv2.matchTemplate(grayimage, cv2.cvtColor(templates[i], cv2.COLOR_BGR2GRAY), cv2.TM_CCOEFF)
#         min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
#
#         if max_val > 50000000:
#             print("Found Face: {}".format(files[i]))

def get_image_from_S3(fileName):

    obj = get_s3().Bucket('multimedia-term-project').Object(fileName)

    buffer = obj.get()["Body"].read()

    nparr = numpy.fromstring(buffer, numpy.ubyte)
    img_np = cv2.imdecode(nparr, cv2.COLOR_BGR2GRAY)
    
    find_faces(img_np, fileName)

keepGoing = True
i = 0
while keepGoing and i < 30:
    try:
        print("Connection try {}".format(i))
        connectionParams = pika.ConnectionParameters(host="rabitmq")
        connection = pika.BlockingConnection(connectionParams)
        channel = connection.channel()
        keepGoing = False
    except pika.exceptions.ConnectionClosed:
        time.sleep(1)
        i = i + 1


def callback(ch, method, properties, body):
    image = json.load(body)
    get_image_from_S3(image["name"])


channel.basic_consume(callback,
                      queue='images',
                      no_ack=True)

print('Waiting for messages. To exit press CTRL+C')
channel.start_consuming()