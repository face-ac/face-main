#BN imports
import face_recognition
import glob
import os
import sys

#AV imports
import cv2
import time

#path to main folder
root = os.path.dirname(os.path.abspath('machine learning facial recognition'))
#path subject to change on final product
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#hold .jpg names to be called in later functions
known_files = []
#unknown_files = []

# fills known_files with .jpg file names found in folder 'known'
for fileName_relative in glob.glob(root+"./known/*.jpg", recursive = True):

    fileName_absolute = os.path.basename(fileName_relative)                

    print("Only file name: ", fileName_absolute)
    #type(fileName_absolute)
    known_files.append(fileName_absolute)

#Aakash's code here
# the value 0 states that we are using integrated camera.
# If we cannot detect the camera module, might wanna change this parameter
captured_image = cv2.VideoCapture(0)
result = True

while result:

    ret, img = captured_image.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Draw the rectangle around each face
    count = 0

    for (x,y,w,h) in faces:

        face = img[y:y+h, x:x+w] 
        cv2.imwrite(str(count)+'.jpg', face)



        for files in known_files:
    		print(files)
    		#load face in known_image
    		known_image = face_recognition.load_image_file("./known/"+files)
    		#load unknown image 						jpg file name subject to change, may need to be altered
    		unknown_image = face_recognition.load_image_file(str(count) + ".jpg")
    		#load known features
    		known_encoding = face_recognition.face_encodings(known_image)[0]
    		#load unknown features
    		unknown_encoding = face_recognition.face_encodings(unknown_image)[0]
    		#compare features
    		results = face_recognition.compare_faces([known_encoding], unknown_encoding)
    		#print(results)
    		if(results[0] == True):
        		print('yes')
        		#TODO unlock door function here (Jared go here)
        
    		#TODO delete/move jpg from folder to backend (for logging purposes) (Ben here)


        count+=1
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

time.sleep(3)
captured_image.release()
cv2.destroyAllWindows()

