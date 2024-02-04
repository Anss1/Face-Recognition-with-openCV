import cv2
import string
from glob import glob
from gtts import gTTS
from multiprocessing import Pool
from scipy.spatial import distance as dist
import define_constants as const
import os
import smtplib
import imghdr
from email.message import EmailMessage
import RPi.GPIO as GPIO
import time







# Define helper functions
def get_num_of_images(path):
    return len(glob(path[0]+'/*.*'))


def get_names(path):
    name = path.split(os.sep)[-1].split('.')[0]
    name = string.capwords(name.replace("_", " "))
    return name

def get_images(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return img

def get_EAR_ratio(eye_points):
    # euclidean distance between two vertical eye landmarks
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])

    # euclidean distance between horizontal eye landmarks
    C = dist.euclidean(eye_points[0], eye_points[3])

    # Eye Aspect Ratio
    return (A + B) / (2.0 * C)

def text_to_speech(text):
    # Text to Sppech
    gtts_obj = gTTS(text=text, lang='en', slow=False)
    gtts_obj.save('assets/text_to_speech/text_to_speech.mp3')

    mixer.init()
    mixer.music.load('assets/text_to_speech/text_to_speech.mp3')
    mixer.music.play()

    
def send_emergency_Email():
    
    Sender_Email = "facerecognition342@gmail.com"
    Owner_Email = "anselpop34@gmail.com"
    Password = "Mohamed999888"

    newMessage = EmailMessage()                         
    newMessage['Subject'] = "Home Security!" 
    newMessage['From'] = Sender_Email                   
    newMessage['To'] = Owner_Email                   
    newMessage.set_content('Unknown person is knocking the door!')

    with open(f"{const.Unknown_people}/Unknown_person.jpg", 'rb') as f:
        image_data = f.read()
        image_type = imghdr.what(f.name)
        image_name = f.name
    newMessage.add_attachment(image_data, maintype='image', subtype=image_type, filename=image_name)

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
        
        smtp.login(Sender_Email, Password)              
        smtp.send_message(newMessage)
        print("Message sent")



def distance():
    #GPIO Mode (BOARD / BCM)
    GPIO.setmode(GPIO.BCM)
     
    #set GPIO Pins
    GPIO_TRIGGER = 20
    GPIO_ECHO = 21


    GPIO.setwarnings(False)
    #set GPIO direction (IN / OUT)
    GPIO.setup(GPIO_TRIGGER, GPIO.OUT)
    GPIO.setup(GPIO_ECHO, GPIO.IN)
    
    # set Trigger to HIGH
    GPIO.output(GPIO_TRIGGER, True)
 
    # set Trigger after 0.01ms to LOW
    time.sleep(0.00001)
    GPIO.output(GPIO_TRIGGER, False)
 
    StartTime = time.time()
    StopTime = time.time()
 
    # save StartTime
    while GPIO.input(GPIO_ECHO) == 0:
        StartTime = time.time()
 
    # save time of arrival
    while GPIO.input(GPIO_ECHO) == 1:
        StopTime = time.time()
 
    # time difference between start and arrival
    TimeElapsed = StopTime - StartTime
    # multiply with the sonic speed (34300 cm/s)
    # and divide by 2, because there and back
    distance = (TimeElapsed * 34000) / 2
 
    return distance
