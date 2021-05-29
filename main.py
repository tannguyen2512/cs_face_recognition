# Import libraries
import cv2
from matplotlib import pyplot as plt
from datetime import datetime
import sys
import os

# CONSTANTS
IMG_WIDTH, IMG_HEIGHT = 416, 416
if len(sys.argv) >= 2:
    DATA_FOLDER = os.path.join('./data', sys.argv[1])
    print('Any captured frames are saved in: ',DATA_FOLDER)
    os.makedirs(DATA_FOLDER, exist_ok=True)
SAVE_FORMAT = '.jpg'

def import_yolo():
    ''' 
    Import the pre-trained model
    '''
    MODEL = './yolo/yolov3-face.cfg'
    WEIGHT = './yolo/yolov3-wider_16000.weights'

    net = cv2.dnn.readNetFromDarknet(MODEL, WEIGHT)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return net

def yolo_forward_pass(frame):
    '''
    Pass each captured frame to the model
    '''
    # Making blob object from original image
    blob = cv2.dnn.blobFromImage(
        frame,
        1/255,
        (IMG_WIDTH, IMG_HEIGHT),
        [0,0,0],
        1,
        crop = False
    )

    #  Set model input
    net.setInput(blob)

    # Define the layers that we want to get the outputs from
    output_layers = net.getUnconnectedOutLayersNames()

    # Run 'prediction
    outs = net.forward(output_layers)
    return outs

def find_high_confidence_bounding_boxes(outs):
    '''
    Scan through all the bounding boxes output from the network and keep only
    the ones with high confidence scores. Assign the box's class label as the
    class with the highest score.
    '''
    confidences = []
    boxes = []

    # Each frame produces 3 outs corresponding to 3 output layers
    for out in outs:
        # One out has multiple predictions for multiple captured objects.
        for detection in out:
            confidence = detection[-1]
            # Extract position data of face area (only area with high confidence)
            if confidence > 0.5:
                center_x = int(detection[0] * frame_width)
                center_y = int(detection[1] * frame_height)
                width = int(detection[2] * frame_width)
                height = int(detection[3] * frame_height)
                
                # Find the top left point of the bounding box 
                topleft_x = center_x - (width//2)  
                topleft_y = center_y - (height//2) 
                confidences.append(float(confidence))
                boxes.append([topleft_x, topleft_y, width, height])

    # Perform non-maximum suppression to eliminate 
    # redundant overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return indices, boxes, confidences

net = import_yolo()

# Open the webcam, show its captured images continuously
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

while True:
    ret, frame = cap.read()
    #  frame is now the image capture by the webcam (one frame of the video)
    # cv2.imshow('Input', frame)

    outs = yolo_forward_pass(frame)

    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    
    (indices, boxes, confidences) = find_high_confidence_bounding_boxes(outs)

    result = frame.copy()
    final_boxes = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        final_boxes.append(box)

        # Extract position data
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]

        # Draw bouding box with the above measurements
        ### YOUR CODE HERE
        cv2.rectangle(
            result,
            (left, top),
            (left + width, top + height),
            (0,255,0),
            1
        )
            
        # Display text about confidence rate above each box
        text = f'{confidences[i]:.2f}'
        ### YOUR CODE HERE
        cv2.putText(
            result,
            text,
            (left, top - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0,255,0),
            1, 
            cv2.LINE_AA
        )
    text = f'Number of faces detected: {len(indices)}'    
    cv2.putText(
            result,
            text,
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0,255,0),
            1, 
            cv2.LINE_AA
        )
    
    cv2.imshow('face detection', result)

    c = cv2.waitKey(1)
    
    if c == ord('s') :   # Save the capture frame when pressing SPACE
        fname = os.path.join(DATA_FOLDER, datetime.today().strftime('%y%m%d%H%M%S')+SAVE_FORMAT)
        print('Save captured frame as ', fname)
        cv2.imwrite(fname, frame)
    elif c == 27 :    # Break when pressing ESC
        break

cap.release()
cv2.destroyAllWindows()