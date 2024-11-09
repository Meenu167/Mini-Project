import cv2
import numpy as np
from .extract_i import slant_angle, pos_of_dot, height_of_dot
import os
import pickle
import math
from django.conf import settings


# image will be the scanned BGR image in type np.array with dimensions (height, width, 3)
# image = cv2.imread("F:\Djnago\graphology\graphapp\extract\letter.png")
# print(image)

def predict(image):
    slant = slant_angle(image)
    pos = pos_of_dot(image)
    print(pos)
    height = height_of_dot(image)

    if height > 30:
        low = 0
        high = 1
    else:
        low = 1
        high = 0

    r = l = b = 0
    if pos == "Right":
        r = 1
    elif pos == "Left":
        l = 1
    else:
        b = 1

    features1 = [[l, b, r]]
    features2 = [[slant, height, high, low]]
    print(features1)
    with open('graphapp/extract/models/model_personality', 'rb') as f:
        model_personality = pickle.load(f)

    with open('graphapp/extract/models/model_behaviour', 'rb') as f:
        model_behavior = pickle.load(f)

    pred1 = model_behavior.predict(features1)
    pred2 = model_personality.predict(features2)

    print("Behavior Type: ", pred1[0])
    print("Personality Type: ", pred2[0])

    confidence1 = model_behavior.decision_function(features1)
    confidence2 = model_personality.decision_function(features2)

    probability1 = 1 / (1 + np.exp(-confidence1))
    probability2 = 1 / (1 + np.exp(-confidence2))

    print("Probability of classification: ", probability1)
    print("Probability of classification: ", probability2)

    print("Probablity of being Outgoing and Expressive :",
        round(probability1[0][0]*100), "%")
    print("Probablity of being Rational: ", round(probability1[0][1]*100), "%")
    print("Probablity of being Shy and Reserved : ",
        round(probability1[0][2]*100), "%")

    print("Probablity of being Balanced :", round(probability2[0][0]*100), "%")
    print("Probablity of being Extroverted: ", round(probability2[0][1]*100), "%")
    print("Probablity of being Introverted : ", round(probability2[0][2]*100), "%")
    
    if pred1[0] == 'Shy&Reserved':
        pred1[0] = 'More reflective and introspective'

    elif pred1[0] == 'Outgoing&Expressive':
        pred1[0] = 'Optimistic, Forward-looking and Future-oriented Mindset'

    elif pred1[0] == 'Rational':
        pred1[0] = 'Attention to detail'

    return {
        'behaviour_type':pred1[0],
        'perosnality_type':pred2[0],
        'behaviour_probability':{
            'outgoing':round(probability1[0][0]*100),
            'rational':round(probability1[0][1]*100),
            'shy':round(probability1[0][2]*100),
        },
        "personality_probability":{
            'balanced':round(probability2[0][0]*100),
            'extroverted':round(probability2[0][1]*100),
            'introverted':round(probability2[0][2]*100),
        }
    }
#predict(image)
