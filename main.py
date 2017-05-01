import json
import os
import time
import codecs


import cv2
import numpy
import boto3
import botocore
import pika
import redis


"""
Redis
"""
r = redis.StrictRedis(host="192.168.99.100")#redis


def get(key):
    return list(str(r.get(key)).split(" ")[1:])

"""
S3
"""
def get_s3():
    cred = json.load(open("aws.config.json"))
    return boto3.resource('s3', aws_access_key_id=cred["accessKeyId"], aws_secret_access_key=cred["secretAccessKey"],
                        config=botocore.config.Config("us-east-2"))


def get_image_from_s3(fileName):
    obj = get_s3().Bucket('multimedia-term-project').Object(fileName)

    buffer = obj.get()["Body"].read()

    nparr = numpy.fromstring(buffer, numpy.ubyte)
    img_np = cv2.imdecode(nparr, cv2.COLOR_BGR2GRAY)
    return img_np


def put_image(image):
    cv2.imwrite(image["name"], image["face"])
    im = open(image["name"], 'rb')
    get_s3().Bucket('multimedia-term-project').Object(image["name"]).put(Body=im, ACL='public-read')
    im.close()
    os.remove(image["name"])


"""
Faces
"""
def get_faces (userid):
    faceids = get(userid)
    faces = []
    for faceid in faceids:
        faces.append({"name": faceid, "face": get_image_from_s3(faceid)})
    return None if faces == [] else faces


def find_faces(image, image_data):
    print(image_data["name"])
    r.set(image_data["name"], "")
    
    faceCascade = cv2.CascadeClassifier("haarcascade_profileface.xml")
    grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(grayimage)
    faceimages = []
    
    for (x, y, w, h) in faces:
        faceimages.append({"face": image[y:y+h, x:x+w], "name": "{}_faces_{}.jpeg".format(image_data["name"], y)})
    
    return faceimages


def template_match(image, template):
    grayimage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    graytemplate = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(grayimage, graytemplate, cv2.TM_CCOEFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return max_val


def match_faces(image, image_data, faces):
    userFaces = get_faces(image_data["userId"])
    for face in faces:
        if userFaces is not None:
            face_score = 0
            face_scores = {}
            for userFace in userFaces:
                score = template_match(face["face"], userFace["face"])
                face_scores[score] = userFace
                face_score = max(score, face_score)

            if face_score > 50_000_000_000:
                r.append(image_data["name"], face_scores[face_score]["name"])
                continue

        r.append(image_data["userId"], face["name"])
        r.append(image_data["name"], face["name"])
        put_image(face)

"""
Rabbitmq
"""
keepGoing = True
i = 0
while keepGoing and i < 30:
    try:
        print("Connection try {}".format(i))
        connectionParams = pika.ConnectionParameters(host="192.168.99.100") #rabbitmq
        connection = pika.BlockingConnection(connectionParams)
        channel = connection.channel()
        keepGoing = False
    except pika.exceptions.ConnectionClosed:
        time.sleep(1)
        i = i + 1


def callback(ch, method, properties, body):
    image_data = json.loads(str(body)[2:-1])
    print("Name: {name}".format(**image_data))
    
    image = get_image_from_s3(image_data["name"])
    faces = find_faces(image, image_data)
    match_faces(image, image_data, faces)
    

channel.queue_declare(queue='images')
channel.basic_consume(callback, queue='images', no_ack=True)

print('Waiting for messages. To exit press CTRL+C')
channel.start_consuming()
