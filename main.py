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



INFO = "info"
ERROR = "error"

def log(message: str, fields: dict, level=INFO) -> None:
    """Basic logger.
    :param message: short message to summarize the log statement
    :param fields: any detailed fields to send
    :param level: INFO or ERROR
    :return: None
    """
    if fields and not isinstance(fields, dict):
        raise Exception("fields must be of type dict")

    data = {}
    if fields:
        data = fields

    data["level"] = level
    data["message"] = message
    data["ts"] = time.time()

    print(data)

    _send(data)


def _send(data: dict) -> requests.Response:
    """Raw HTTP Post request to the logging API Endpoint.
    For internal use only. Don't call this func directly.
    :param data: the body of the request
    :return: requests.Response
    """
    res = None
    try:
        res = requests.post(
            url=API_ENDPOINT, data=data, headers={"Authorization": API_TOKEN}
        )
    except Exception as e:
        print("error sending log", e)
    finally:
        return res



#TODO import C files for lcd?




#path to main folder
root = os.path.dirname(os.path.abspath('face-main'))
#path subject to change on final product
face_cascade = cv2.CascadeClassifier('opencv-master/data/haarcascades/haarcascade_frontalface_default.xml')

#hold .jpg names to be called in later functions
known_files = []
#unknown_files = []

# fills known_files with .jpg file names found in folder 'known'
for fileName_relative in glob.glob(root+"./known/*.jpg", recursive = True):

    fileName_absolute = os.path.basename(fileName_relative)                

    print("Only file name: ", fileName_absolute)
    #type(fileName_absolute)
    known_files.append(fileName_absolute)

for fileName_relative in glob.glob(root+"./known/*.jpeg", recursive = True):

    fileName_absolute = os.path.basename(fileName_relative)                

    print("Only file name: ", fileName_absolute)
    #type(fileName_absolute)
    known_files.append(fileName_absolute)




while 1:
    captured_image = cv2.VideoCapture(0)
    
    ret, img = captured_image.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    count = 0
    
    for (x,y,w,h) in faces:
        face = img[y:y+h, x:x+w]
        os.chdir(root+"./logger")
        cv2.imwrite(str(count)+'.jpg', face)
        os.chdir(root)
        
        for files in known_files:
            print(files)
            known_image = face_recognition.load_image_file("./known/"+files)
            unknown_image = face_recognition.load_image_file("./logger/"+str(count) + ".jpg")
            height, width, _ = unknown_image.shape
            face_location = (0, width, height, 0)
            known_encoding = face_recognition.face_encodings(known_image)[0]
            unknown_encoding = face_recognition.face_encodings(unknown_image, known_face_locations=[face_location])[0]
            results = face_recognition.compare_faces([known_encoding], unknown_encoding)
            if(results[0] == True):
                print('face found')
                #log_packet1 = dict(name = files, image = face)
                #log('Face Recognized', log_packet1, INFO)
                #unlock door
                time.sleep(7) #wait 7s
                #lock door
        count+=1
        #log_packet2 = dict(name = 'unknown', image = face)
        #log('Unrecognized face', log_packet2, INFO)
    time.sleep(3)
    captured_image.release()
    cv2.destroyAllWindows()
    
    
    
    
    
    
    
    
    
    
    
    
    
    