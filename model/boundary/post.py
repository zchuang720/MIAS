import requests
import base64
import json
import time
import hashlib
import threading
import copy
import cv2

"""
[
    {"desc":"",
    "image":"base64",
    "type":"人/动物/车辆/其他",
    "position":[1,2,3,4]
    },
]
"""
result_dict = {
    "desc": "",
    "image": "",
    "type": "",
    "position": [],
}
