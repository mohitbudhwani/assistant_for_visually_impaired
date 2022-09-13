# Detecting Objects in Real Time with OpenCV deep learning library
#
# Algorithm:
# Reading stream video from camera --> Loading YOLO v3 Network -->
# --> Reading frames in the loop --> Getting blob from the frame -->
# --> Implementing Forward Pass --> Getting Bounding Boxes -->
# --> Non-maximum Suppression --> Drawing Bounding Boxes with Labels -->
# --> Showing processed frames in OpenCV Window
#
# Result:
# Window with Detected Objects, Bounding Boxes and Labels in Real Time

# Press v to get audio feedback of the current frame
# Press q to quit realtime object detection

# Importing needed libraries
import numpy as np
import cv2
import time
import engineio
import pyttsx3
import math
import argparse
import keyboard
import msvcrt


def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn=frame.copy()
    frameHeight=frameOpencvDnn.shape[0]
    frameWidth=frameOpencvDnn.shape[1]
    blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    detections=net.forward()
    faceBoxes=[]
    for i in range(detections.shape[2]):
        confidence=detections[0,0,i,2]
        if confidence>conf_threshold:
            x1=int(detections[0,0,i,3]*frameWidth)
            y1=int(detections[0,0,i,4]*frameHeight)
            x2=int(detections[0,0,i,5]*frameWidth)
            y2=int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            #cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn,faceBoxes


def speak(text):
    engineio = pyttsx3.init()
    engineio.say(text)
    engineio.runAndWait()


"""
Start of:
Reading stream video from camera
"""
def yo():

    parser=argparse.ArgumentParser()
    parser.add_argument('--image')

    args=parser.parse_args()

    faceProto="opencv_face_detector.pbtxt"
    faceModel="opencv_face_detector_uint8.pb"
    ageProto="age_deploy.prototxt"
    ageModel="age_net.caffemodel"
    genderProto="gender_deploy.prototxt"
    genderModel="gender_net.caffemodel"

    MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
    ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList=['Male','Female']

    faceNet=cv2.dnn.readNet(faceModel,faceProto)
    ageNet=cv2.dnn.readNet(ageModel,ageProto)
    genderNet=cv2.dnn.readNet(genderModel,genderProto)
    # Defining 'VideoCapture' object
    # and reading stream video from camera
    camera = cv2.VideoCapture(0)

    # Preparing variables for spatial dimensions of the frames
    h, w = None, None

    with open('yolo-coco-data/coco.names') as f:
        labels = [line.strip() for line in f]


# # Check point
# print('List with labels names:')
# print(labels)

# Loading trained YOLO v3 Objects Detector
# with the help of 'dnn' library from OpenCV
# Pay attention! If you're using Windows, yours paths might look like:
# r'yolo-coco-data\yolov3.cfg'
# r'yolo-coco-data\yolov3.weights'
# or'
# 'yolo-co:
# 'yolo-coco-data\\yolov3.cfgco-data\\yolov3.weights'
    network = cv2.dnn.readNetFromDarknet('yolo-coco-data/yolov3.cfg',
                                     'yolo-coco-data/yolov3.weights')
    
    # For using GPU *******************************************************************************************
    # network.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    network.setPreferableTarget(cv2.dnn.DNN_TARGET_OPENCL_FP16)
    

# Getting list with names of all layers from YOLO v3 network
    layers_names_all = network.getLayerNames()

# # Check point
# print()
# print(layers_names_all)

# Getting only output layers' names that we need from YOLO v3 algorithm
# with function that returns indexes of layers with unconnected outputs
    layers_names_output = \
        [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]

# # Check point
# print()
# print(layers_names_output)  # ['yolo_82', 'yolo_94', 'yolo_106']

# Setting minimum probability to eliminate weak predictions
    probability_minimum = 0.5

# Setting threshold for filtering weak bounding boxes
# with non-maximum suppression
    threshold = 0.3

# Generating colours for representing every detected object
# with function randint(low, high=None, size=None, dtype='l')
    colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

# # Check point
# print()
# print(type(colours))  # <class 'numpy.ndarray'>
# print(colours.shape)  # (80, 3)
# print(colours[0])  # [172  10 127]


    str3=""
    padding=20

    while True:

        voice=False

        if keyboard.is_pressed('v'):
            voice=True
    # Capturing frame-by-frame from camera
        _, frame = camera.read()
    

    # Getting spatial dimensions of the frame
    # we do it only once from the very beginning
    # all other frames have the same dimension
        if w is None or h is None:
        # Slicing from tuple only first two elements
            h, w = frame.shape[:2]

        # BlobfromImage is used for image processing by performing mean subtraction, scaling and optionially channel swapping  Initial Size was (416,416)
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (320, 320),
                                 swapRB=True, crop=False)

    
        network.setInput(blob)  
        start = time.time()
        output_from_network = network.forward(layers_names_output)
        end = time.time()

    
        print('Current frame took {:.5f} seconds'.format(end - start))


        bounding_boxes = []
        confidences = []
        class_numbers = []

    # Going through all output layers after feed forward pass
        for result in output_from_network:
        # Going through all detections from current output layer
            for detected_objects in result:
            # Getting 80 classes' probabilities for current detected object
                scores = detected_objects[5:]
            # Getting index of the class with the maximum value of probability
                class_current = np.argmax(scores)
            # Getting value of probability for defined class
                confidence_current = scores[class_current]

            # # Check point
            # # Every 'detected_objects' numpy array has first 4 numbers with
            # # bounding box coordinates and rest 80 with probabilities
            # # for every class
            # print(detected_objects.shape)  # (85,)

            # Eliminating weak predictions with minimum probability
                if confidence_current > probability_minimum:
                # Scaling bounding box coordinates to the initial frame size
                # YOLO data format keeps coordinates for center of bounding box
                # and its current width and height
                # That is why we can just multiply them elementwise
                # to the width and height
                # of the original frame and in this way get coordinates for center
                # of bounding box, its width and height for original frame
                    box_current = detected_objects[0:4] * np.array([w, h, w, h])

                # Now, from YOLO data format, we can get top left corner coordinates
                # that are x_min and y_min
                    x_center, y_center, box_width, box_height = box_current
                    x_min = int(x_center - (box_width / 2))
                    y_min = int(y_center - (box_height / 2))

                # Adding results into prepared lists
                    bounding_boxes.append([x_min, y_min,
                                       int(box_width), int(box_height)])
                    confidences.append(float(confidence_current))
                    class_numbers.append(class_current)

        results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                               probability_minimum, threshold)

   

    
        if len(results) > 0:
            for i in results.flatten():
            
                x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
                box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

            
                colour_box_current = colours[class_numbers[i]].tolist()

           
                cv2.rectangle(frame, (x_min, y_min),
                          (x_min + box_width, y_min + box_height),
                          colour_box_current, 2)
            
                str1=""
                if 142<=x_min<=300:
                    str1=str1+"center"
                if x_min+box_width<165 or x_min<142:
                    str1=str1+"left"
                if x_min>300:
                    str1=str1+"right"
            
                text_box_current = '{}: {:.4f},'.format(labels[int(class_numbers[i])],
                                                   confidences[i])+str1
            
           
            
                if str(labels[int(class_numbers[i])])=="person":
                    resultImg,faceBoxes=highlightFace(faceNet,frame)
                    for faceBox in faceBoxes:
                        face=frame[max(0,faceBox[1]-padding):
                        min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                        :min(faceBox[2]+padding, frame.shape[1]-1)]

                        blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
                        genderNet.setInput(blob)
                        genderPreds=genderNet.forward()
                        gender=genderList[genderPreds[0].argmax()]
                
                        ageNet.setInput(blob)
                        agePreds=ageNet.forward()
                        age=ageList[agePreds[0].argmax()]
                        cv2.putText(resultImg, f'{gender}, {age}', (x_min,y_min-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)
                        str2="Detected"+f'{gender}'+"with age range"+f'{age}'+"at"+str1
                        if voice==True: #str3!=str2 and 
                            speak(str2)
                            #str3=str2 
                    frame=resultImg
            
                else:
                    cv2.putText(frame, text_box_current, (x_min, y_min - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)
                    str2="Detected"+str(labels[int(class_numbers[i])])+"at"+str1
                    if voice==True:
                        speak(str2)
                        #str3=str2 
            voice=False
                    



   
        cv2.namedWindow('YOLO v3 Real Time Detections', cv2.WINDOW_NORMAL)
        cv2.imshow('YOLO v3 Real Time Detections', frame)
   


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    


    camera.release()
    cv2.destroyAllWindows()


def main():
    yo()

# Checking if current namespace is main, that is file is not imported
if __name__ == '__main__':
    # Implementing main() function
    main()

