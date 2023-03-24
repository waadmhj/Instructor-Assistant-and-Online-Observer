#import libraries
import cv2
import dlib
from scipy.spatial import distance
import mediapipe as mp
import numpy as np
from datetime import datetime


#threshold for object detector 
thres = 0.60
#read from camera
image_pose = cv2.VideoCapture(0)

        
#eye moiton 
def calculate_EAR(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear_aspect_ratio = (A+B)/(2.0*C)
	return ear_aspect_ratio\
               
#sleepy feature
hog_face_detector = dlib.get_frontal_face_detector()
dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")


# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose
# Setup the Pose function for images - independently for the images standalone processing.
pose_image = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
# Setup the Pose function for videos - for video processing.
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.7,min_tracking_confidence=0.7)
# Initialize mediapipe drawing class - to draw the landmarks points.
mp_drawing = mp.solutions.drawing_utils


#--------------------object detection part----------------#
classNames= []
classFile = 'coco.names'
with open(classFile,'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')
 
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

t = datetime.now()
while True:
    try:
        #now = datetime.now()
        suc, frame = image_pose.read()
        image = frame.copy()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = hog_face_detector(gray)
        count = 0
        
        for face in faces:

            face_landmarks = dlib_facelandmark(gray, face)
            leftEye = []
            rightEye = []
            
            for n in range(36,42):
                    x = face_landmarks.part(n).x
                    y = face_landmarks.part(n).y
                    leftEye.append((x,y))
                    next_point = n+1
                    if n == 41:
                            next_point = 36
                    x2 = face_landmarks.part(next_point).x
                    y2 = face_landmarks.part(next_point).y
                    cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

            for n in range(42,48):
                    x = face_landmarks.part(n).x
                    y = face_landmarks.part(n).y
                    rightEye.append((x,y))
                    next_point = n+1
                    if n == 47:
                            next_point = 42
                    x2 = face_landmarks.part(next_point).x
                    y2 = face_landmarks.part(next_point).y
                    cv2.line(frame,(x,y),(x2,y2),(0,255,0),1)

            left_ear = calculate_EAR(leftEye)
            right_ear = calculate_EAR(rightEye)

            EAR = (left_ear+right_ear)/2
            EAR = round(EAR,2)
            if EAR<0.16:
                    now = datetime.now()
                    current_time = now.strftime("%H:%M:%S")
                    cv2.putText(frame,"",(20,100),
                            cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),4)
                    cv2.putText(frame,"Are you Sleepy?",(20,400),
                            cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)
                    delta = datetime.now()-t
                    if delta.seconds >= 3:
                            current_time = now.strftime("%H:%M:%S")
                            print("sleepy",current_time)
                            t = datetime.now()


        classIds, confs, bbox = net.detect(frame,confThreshold=thres)

        
        if len(classIds) != 0:
            for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
                #only draw bounding box over the cell phone and the person 
                if classNames[classId-1] == 'cell phone' or classNames[classId-1] == 'person':
                        cv2.rectangle(frame,box,color=(0,255,0),thickness=2)
                        cv2.putText(frame,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                        cv2.putText(frame,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                        cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)
                else:
                        continue


        #print detecting cell phone
        if len(classIds) != 0:
            for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
                if classNames[classId-1] == 'cell phone':
                        now = datetime.now()
                        delta = datetime.now()-t
                        if delta.seconds >= 3:
                                current_time = now.strftime("%H:%M:%S")
                                print("Cell phone DETECTED!, at Time = ", current_time)
                                t = datetime.now()
                        

        
        #counting number of persons in the frame
        count = 0
        if len(classIds) != 0:
            for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
                if classNames[classId-1] == 'person':
                        count+=1
                        if(count>1):
                                now = datetime.now()
                                delta = datetime.now()-t
                                if delta.seconds >= 3:
                                        current_time = now.strftime("%H:%M:%S")
                                        print("another PERSON detected!, at Time = ", current_time)
                                        t = datetime.now()


    
        cap_in_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultant = pose_video.process(cap_in_RGB)
        
        # Draw pose:
        mp_drawing.draw_landmarks(image=frame, landmark_list=resultant.pose_landmarks, connections=mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=3,
                                                                               circle_radius=3),
                                  connection_drawing_spec=mp_drawing.DrawingSpec(color=(49, 125, 237), thickness=2,
                                                                                 circle_radius=2))
        h, w = frame.shape[:2]
        lm = resultant.pose_landmarks

        avgw = (int(lm.landmark[11].x * w) + int(lm.landmark[12].x * w)) / 2
        avgh = (int(lm.landmark[11].x * h) + int(lm.landmark[12].x * h)) / 2

        avgw1 = round(avgw)
        avgh1 = round(avgh)

        point1 = np.array([avgw1, avgh1])
        point2 = np.array((int(lm.landmark[0].x * w), int(lm.landmark[0].y * h)))

        # calculating Euclidean distance
        dist = np.linalg.norm(point2 - point1)

        cv2.imshow('Cap Pose', frame)
        cv2.waitKey(30)

        if dist > 50:
            now = datetime.now()
            delta = datetime.now()-t
            if delta.seconds >= 3:
                current_time = now.strftime("%H:%M:%S")
                print("Student not giving attenion, at Time = ", current_time)
                t = datetime.now()
             
        else:
            continue
        
#if student is outside the frame, show (student missing) 
    except:
        now = datetime.now()
        delta = datetime.now()-t
        if delta.seconds >= 3:
                current_time = now.strftime("%H:%M:%S")
                cv2.putText(frame,"Studnet Missing",(20,400),
                                cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),4)
                #cv2.imshow('Cap Pose', frame)
                cv2.waitKey(30)
                print('Studnet Missing, at Time = ',current_time)
                # Update 't' variable to new time
                t = datetime.now()
        
