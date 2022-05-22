# import necessary packages
from imutils.video import VideoStream
import numpy as np
from imutils.video import FPS
import imutils
import time
import cv2
import urllib
from keras.models import load_model

# initialize the list of class labels MobileNet SSD was trained to detect
# generate a set of bounding box colors for each class
CLASSES = ['background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
#CLASSES = ['motorbike', 'person']
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe('motorbike.prototxt.txt', 'motorbike.caffemodel')

print('Loading helmet model...')
loaded_model = load_model('helmet-nonhelmet_cnn.h5')
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

# initialize the video stream,
print("[INFO] starting video stream...")

def helmet_or_nohelmet(helmet_roi):
    try:
        helmet_roi = cv2.resize(helmet_roi, (224, 224))
        helmet_roi = np.array(helmet_roi,dtype='float32')
        helmet_roi = helmet_roi.reshape(1, 224, 224, 3)
        helmet_roi = helmet_roi/255.0
        return int(loaded_model.predict(helmet_roi)[0][0])
    except:
            pass

# time.sleep(2.0)

# Starting the FPS calculation
fps = FPS().start()

# loop over the frames from the video stream
# i = True
while True:
    # i = not i
    # if i==True:
    rider_without_helmet = False
    try:
        # grab the frame from the threaded video stream and resize it
        # to have a maxm width and height of 600 pixels
        # Loading the video file
        # ret, frame = cap.read()
        req = urllib.request.urlopen('http://192.168.0.185/jpg')
        arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        # resizing the images
        frame = cv2.imdecode(arr, -1)
        frame = imutils.resize(frame, width=800, height=800)

        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        
        # Resizing to a fixed 300x300 pixels and normalizing it.
        # Creating the blob from image to give input to the Caffe Model
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (800, 800)), 0.007843, (800, 800), 127.5)

        # pass the blob through the network and obtain the detections and predictions
        net.setInput(blob)

        detections = net.forward()  # getting the detections from the network
        
        persons = []
        person_roi = []
        motorbi = []
        
        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence associated with the prediction
            confidence = detections[0, 0, i, 2]
            
            # filter out weak detections by ensuring the confidence
            # is greater than minimum confidence
            if confidence > 0.5:
                
                # extract index of class label from the detections
                idx = int(detections[0, 0, i, 1])
                
                if idx == 15:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    # roi = box[startX:endX, startY:endY/4] 
                    # person_roi.append(roi)
                    persons.append((startX, startY, endX, endY))

                if idx == 14:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    motorbi.append((startX, startY, endX, endY))

        xsdiff = 0
        xediff = 0
        ysdiff = 0
        yediff = 0
        p = ()
        
        for i in motorbi:
            mi = float("Inf")
            for j in range(len(persons)):
                xsdiff = abs(i[0] - persons[j][0])
                xediff = abs(i[2] - persons[j][2])
                ysdiff = abs(i[1] - persons[j][1])
                yediff = abs(i[3] - persons[j][3])

                if (xsdiff+xediff+ysdiff+yediff) < mi:
                    mi = xsdiff+xediff+ysdiff+yediff
                    p = persons[j]
                    # r = person_roi[j]


            if len(p) != 0:

                roi = frame[p[1]:p[1]+(p[3]-p[1])//4, p[0]:p[2]]
                helmca = helmet_or_nohelmet(roi)
                # display the prediction
                label = "{}".format(CLASSES[14])
                cv2.rectangle(frame, (i[0], i[1]), (i[2], i[3]), COLORS[14], 2)
                y = i[1] - 15 if i[1] - 15 > 15 else i[1] + 15
                cv2.putText(frame, ['helmet','no-helmet'][helmca], (i[0], y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[14], 2)
                if(helmca == 1):
                    rider_without_helmet = True
                label = "{}".format(CLASSES[15])

                # cv2.rectangle(frame, (p[0], p[1]), (p[2], p[3]), COLORS[15], 2)
                # y = p[1] - 15 if p[1] - 15 > 15 else p[1] + 15

                # if len(roi) != 0:
                #     helmc = helmet_or_nohelmet(roi)
                #     img_array = cv2.resize(roi, (50,50))
                #     gray_img = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
                #     img = np.array(gray_img).reshape(1, 50, 50, 1)
                #     img = img/255.0

    except Exception as e:
        print(e)
        pass
    
    cv2.imshow('Frame', frame)  # Displaying the frame
    key = cv2.waitKey(1) & 0xFF
    if(rider_without_helmet):
        print("NO HELMET SEND SMS")
    time.sleep(1)
    rider_without_helmet = False

    if key == ord('q'): # if 'q' key is pressed, break from the loop
        break
     
    # update the FPS counter
    fps.update()
        

# stop the timer and display FPS information
fps.stop()

print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cv2.destroyAllWindows()
# cap.release()   # Closing the video stream 
