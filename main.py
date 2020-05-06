

#BN imports
import face_recognition
import glob
import os
import sys

#AV imports
import cv2
import time

#BK imports
import requests
from logger import log, INFO

import argparse
#TODO import C files for lcd?




#path to main folder
root = os.path.dirname(os.path.abspath('face-main'))
#path subject to change on final product
face_cascade = cv2.CascadeClassifier('opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')

#hold .jpg names to be called in later functions
known_files = []
#unknown_files = []
    
parser = argparse.ArgumentParser()
#parser.add_argument('--model', help='.tflite model path',
#                        default=os.path.join(default_model_dir,default_model))
#parser.add_argument('--labels', help='label file path',
#                        default=os.path.join(default_model_dir, default_labels))
#parser.add_argument('--top_k', type=int, default=3,
#                        help='number of categories with highest score to display')
parser.add_argument('--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
#parser.add_argument('--threshold', type=float, default=0.1,
#                        help='classifier score threshold')
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
                time.sleep(7) #wait 7s
                #lock door
        count+=1

        if not recognized:
            # Log unrecognized only if we did not recognize it in loop above
            log("unrecognized face", fields={"name": "unknown", "image": face}, level=INFO)

    time.sleep(3)
    captured_image.release()
    cv2.destroyAllWindows()
