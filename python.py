# |||||||||||||||||||||||||||||||||||||||||-- USING WEBCAM (Default) --|||||||||||||||||||||||||||||||||||||||||||||||||||||
#!/usr/bin/env python3
#-->IMPORTING LIBRARIES
import cv2 as cv         #IMPORTING OPENCV MODULE AS cv
import numpy as np       #IMPORTING NUMPY MODULE AS np
import datetime          #IMPORTING DATETIME MODULE to get current Date & Time

#-->OPENCV video capture
FPS = 120
######## EDIT 1 ###########################################
cap = cv.VideoCapture(0,cv.CAP_DSHOW) #captureDevice = camera
#  cv.CAP_DSHOW |This works for OpenCV>=3.4| https://stackoverflow.com/questions/60007427/cv2-warn0-global-cap-msmf-cpp-674-sourcereadercbsourcereadercb-termina
######### EDIT 1 Warning Closed for Webcam  ###############
#For image or Video input |cap = cv.VideoCapture('FilePath')| mp4,jpg etc.,
cap.set(cv.CAP_PROP_FPS, FPS)

######## Edit 2 Change Resolutions ################################################
# cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
# For Getting 4k,2k,1080p,720p,480p,360p,240p,120p Resolution
#   https://stackoverflow.com/questions/19448078/python-opencv-access-webcam-maximum-resolution
############ CLOSED ISSUE ##########################################################

#Constraints
whT = 320  #Weight
confThreshold = 0.5 #0.5 Confidence (Range: 0.1-1.0) #change this for detecting farer/nearner objects clearly.
nmsThreshold = 0.2  #0.2 Non-Max Suppression (Range: 0.1-1.0)

#-->LOADING MODELS
# LOAD MODEL
# Coco Names
classesFile = "C:\\Users\\user1\\Desktop\\Improvised-Object-Detection-using-YOLO-V3\\coco.names"
classNames = [] # List
with open(classesFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
 #---------------------------------------------------||  For CMD Output starts here ||
print('Initializing COCO Dataset Names (.names)...') 
print('Initializing ModelConfiguration (.cfg)...')  
print('Initializing ModelWeights (.weights)...') 
print(' ')    
print(classNames)
print(' ') 
print('Class Type: ',type(classNames)) # LIST []
print('Total Objects: ',len(classNames))  # 80
print(' ') 
print('Initializing new window...')
print(' ') 
print('Object Detection Started...')
print(' ')
#-------------------------------------------------------|| Ends Here ||

#--> Model Files
#For Yolo 320      ||yolov3.weights: 236 MB (24,80,07,048 bytes)|| 
modelConfiguration = "C:\\Users\\user1\\Desktop\\Improvised-Object-Detection-using-YOLO-V3\\yolov3.cfg"
modelWeights = "C:\\Users\\user1\\Desktop\\Improvised-Object-Detection-using-YOLO-V3\\yolov3.weights"
'''
For Yolo yolov3-tiny ||yolov3-tiny.weights: 33.7 MB (3,54,34,956 bytes)||
modelConfiguration = "C:\\Users\\user1\\Desktop\\Improvised-Object-Detection-using-YOLO-V3\\yolov3-tiny.cfg"
modelWeights = "C:\\Users\\user1\\Desktop\\Improvised-Object-Detection-using-YOLO-V3\\yolov3-tiny.weights"

'''
net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)

#-->TO FIND AN OBJECT 
def findObjects(outputs, img):
    hT, wT, cT = img.shape

    bbox = [] #Bounding box
    classIds = []
    confs = []
    for output in outputs:
        for det in output:
            scores = det[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                w, h = int(det[2]*wT), int(det[3]*hT)
                x, y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)
                bbox.append([x, y, w, h])
                classIds.append(classId)
                confs.append(float(confidence))

    indices = cv.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

#-->BOXES FOR DETECTED OBJECTS X,Y,W,H
    for i in indices:
        i = i[0]
        box = bbox[i]
        x, y, w, h = box[0], box[1], box[2], box[3]
        # print(x,y,w,h)
        #DATE & TIME 
        now = datetime.datetime.now()
        cv.putText(img, str(now.strftime("%d-%m-%Y %H:%M:%S")), (10, 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)
        # TITLE 
        cv.putText(img, str('RUNNING : OBJECT DETECTION'), (10, 470),
                   cv.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv.LINE_AA)
        '''Frames Per Second (FPS) ||Use any function for FPS
        cv.putText(img, str('FPS: '), (545, 25),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv.LINE_AA)'''
        #Rectangle for objects detected
        cv.rectangle(img, (x, y), (x+w, y+h), (255,20,147), 2)
        
        '''
        TO SEE ARRAY MATRIX in CMD <class 'numpy.ndarray'>
        ArrayMatrix = cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 255), 2)
        print(ArrayMatrix)
        print(type(ArrayMatrix))
        '''
        cv.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%',
                   (x, y-10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        print('Detected:',classNames[classIds[i]].upper()) # , bbox for Bounding Boxes
        # obj = classNames[classIds[i]].upper()
        # print(obj) 
        # print(type(classIds)) #LIST
        # print(type(obj))     # STR
while True:
    success, img = cap.read()

    blob = cv.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1, crop=False)
        
    net.setInput(blob)
    layersNames = net.getLayerNames()
    outputNames = [(layersNames[i[0] - 1])
                   for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(outputNames)
    findObjects(outputs, img)

    # print('Initializing in a new window...')
    cv.imshow('Improvised Object Detection using YOLOv3', img)    
    # print('Detecting....')
    # print(Counter(blob))
    key = cv.waitKey(20)
    if key == 27:  # exit on ESC
        print(' ')
        print('Window Closed Successfully...')
        break
cap.release()
cv.destroyAllWindows()
exit()
