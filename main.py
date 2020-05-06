

#BN imports
import face_recognition
import glob
import os
import sys

#used for python search path
sys.path.append('/usr/local/lib/python3.7/site-packages')

#AV imports
import cv2
import time

#BK imports
import requests
from logger import log, INFO

#JG imports
import board 
import digitalio
import adafruit_character_lcd.character_lcd as characterlcd
import argparse

#init gpio output for lock
lock = digitalio.DigitalInOut(board.GPIO_P13)
lock.direction = digitalio.Direction.OUTPUT

#init gpio for LCD
lcd_rs = digitalio.DigitalInOut(board.GPIO_P37)
lcd_en = digitalio.DigitalInOut(board.GPIO_P36)
lcd_d7 = digitalio.DigitalInOut(board.GPIO_P16)
lcd_d6 = digitalio.DigitalInOut(board.GPIO_P18)
lcd_d5 = digitalio.DigitalInOut(board.GPIO_P29)
lcd_d4 = digitalio.DigitalInOut(board.GPIO_P31)

#init LCD type
lcd_columns = 16
lcd_rows = 2
lcd = characterlcd.Character_LCD_Mono(lcd_rs, lcd_en, lcd_d4, lcd_d5, lcd_d6, lcd_d7, lcd_columns, lcd_rows)

#path to main folder
root = os.path.dirname(os.path.abspath('face-main'))
#path subject to change on final product
face_cascade = cv2.CascadeClassifier('opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')

#hold .jpg names to be called in later functions
known_files = []
#unknown_files = []
    
parser = argparse.ArgumentParser()
parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
args = parser.parse_args()

# fills known_files with .jpg file names found in folder 'known'
for fileName_relative in glob.glob(root+"/known/*jpg", recursive = True):

    fileName_absolute = os.path.basename(fileName_relative)                

    print("Only file name: ", fileName_absolute)
    #type(fileName_absolute)
    known_files.append(fileName_absolute)

for fileName_relative in glob.glob(root+"/known/*jpeg", recursive = True):

    fileName_absolute = os.path.basename(fileName_relative)                

    print("Only file name: ", fileName_absolute)
    #type(fileName_absolute)
    known_files.append(fileName_absolute)




while 1:
    captured_image = cv2.VideoCapture(args.camera_idx)
    
    ret, img = captured_image.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    count = 0
    
    for (x,y,w,h) in faces:
        face = img[y:y+h, x:x+w]
        os.chdir(root+"/logger")
        cv2.imwrite(str(count)+'.jpg', face)
        os.chdir(root)

        # recognized is a flag for recognized/unrecognized faces.
        # We need this so we don't log "face recognized" and "unrecognized face" for the same face.
        recognized = False
        
        for files in known_files:
            print(files)
            known_image = face_recognition.load_image_file(root+"/known/"+files)
            unknown_image = face_recognition.load_image_file(root+"/logger/"+str(count) + ".jpg")
            height, width, _ = unknown_image.shape
            face_location = (0, width, height, 0)
            known_encoding = face_recognition.face_encodings(known_image)[0]
            unknown_encoding = face_recognition.face_encodings(unknown_image, known_face_locations=[face_location])[0]
            results = face_recognition.compare_faces([known_encoding], unknown_encoding)
            if(results[0] == True):
                # Set recognized flag and log
                recognized = True
                log("face recognized", fields={"name": files, "image": face}, level=INFO)
                #unlock door
                lock.value = False
                lcd.message = "Access Granted"
                time.sleep(7) #wait 7s
                #lock door
                lock.value = True
                lcd.clear()
        count+=1

        if not recognized:
            # Log unrecognized only if we did not recognize it in loop above
            log("unrecognized face", fields={"name": "unknown", "image": face}, level=INFO)
            lcd.message = "Access Denied"
            time.sleep(5) #wait 5s
            lcd.clear()
    time.sleep(3)
    captured_image.release()
    cv2.destroyAllWindows()
