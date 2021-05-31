# Import libraries
import cv2, time, timeit, os, sys
from matplotlib import pyplot as plt
from datetime import datetime
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import threading
import concurrent.futures


# CONSTANTS
IMG_WIDTH, IMG_HEIGHT = 416, 416
if len(sys.argv) >= 2:
    DATA_FOLDER = os.path.join('./data', sys.argv[1])
    print('Any captured frames are saved in: ',DATA_FOLDER)
    os.makedirs(DATA_FOLDER, exist_ok=True)
SAVE_FORMAT = '.jpg'
MODEL_FOLDER = './model'


CLASS_IX = {'Luke': 0, 'Steph': 1, 'Uyen': 2, 'tan': 3}
IX_CLASS = {CLASS_IX[ix]:ix for ix in CLASS_IX}

CONFIDENCE_THRESH = 0.6

############################################
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

def find_high_confidence_bounding_boxes(outs, frame_height, frame_width):
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
                topleft_x = max(center_x - (width//2),0)
                topleft_y = max(center_y - (height//2),0)
                confidences.append(float(confidence))
                boxes.append([topleft_x, topleft_y, width, height])

    # Perform non-maximum suppression to eliminate 
    # redundant overlapping boxes with lower confidences.
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return indices, boxes, confidences

def preprocess_face(face):
    pFace = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    pFace = cv2.resize(
                pFace, 
                (second_layer_input_width, second_layer_input_height), 
                cv2.INTER_AREA)
    pFace_array = image.img_to_array(pFace).astype('float32')/255
    pFace_array = np.expand_dims(pFace_array, axis = 0)
    return pFace_array

def load_compile_model(pretrained_path):
    pretrained_second_layer = load_model(pretrained_path)
    pretrained_second_layer.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics = ['accuracy'])
    return pretrained_second_layer

def detect_predict_face(frame):
    # initiate the internal result variable
    tmp = {
        'topleft': [],
        'botright': [],
        'person': [],
        'confidence': [],
        'text_start': []
    }
    outs = yolo_forward_pass(frame)
    (indices, boxes, confidences) = find_high_confidence_bounding_boxes(outs, frame_height, frame_width)
    
    for i in indices:
        i = i[0]
        box = boxes[i]
        
        # Extract position data
        left, top, width, height = box
        
        # Crop the face
        face = frame[top:(top + height), left: (left + width)]
        pFace_array = preprocess_face(face)
        # Forward pass the pretrained model
        pred_proba = pretrained_second_layer.predict(pFace_array)
        pred_label = np.argmax(pred_proba)
        pred_person = IX_CLASS[pred_label]
        
        # Display text about confidence rate above each box
        if np.max(pred_proba) < CONFIDENCE_THRESH :
            text = f'Unknown {np.max(pred_proba):.4f}'  
            tmp['person'].append(None),
            tmp['confidence'].append(None)  
        else :
            text = f'{pred_person}({pred_label}) {np.max(pred_proba):.4f}'
            tmp['person'].append(pred_person),
            tmp['confidence'].append(np.max(pred_proba))
        tmp['topleft'].append((left, top))
        tmp['botright'].append((left + width, top + height))
        if top - 5 <= 0 :
            tmp['text_start'].append((left, top + height + 15))
        else :
            tmp['text_start'].append((left, top - 5))

    return tmp

############################################
net = import_yolo()

global_cache = {
    'topleft': [],
    'botright': [],
    'person': [],
    'confidence': [],
    'text_start': []
}

pretrained_path = 'tan_face_model.h5'
# pretrained_path = 'facenet_keras.h5'
# pretrained_path = 'steph_model_checkpoint_10.h5'
pretrained_path = os.path.join(MODEL_FOLDER,pretrained_path)
pretrained_second_layer = load_compile_model(pretrained_path)
print('='*20)
print('Using model saved in ', pretrained_path)
print('='*20)

second_layer_input_width = pretrained_second_layer.input_shape[1]
second_layer_input_height = pretrained_second_layer.input_shape[2]

# Open the webcam, show its captured images continuously
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

frame_count = 0
while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]
    
    result = frame.copy()
    
    if frame_count % 15 == 0:
        # cache = detect_predict_face(frame)

        # Run on a new thread with concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future = executor.submit(detect_predict_face, frame)
            global_cache = future.result()
        frame_count = 1  
    else :  
        frame_count += 1

    for i in range(len(global_cache['topleft'])):
        box_start = global_cache['topleft'][i]
        box_end = global_cache['botright'][i]
        text_start = global_cache['text_start'][i]

        if global_cache["person"][i] == None :
            text = 'Unknown'
        else :
            text = f'{global_cache["person"][i]} {global_cache["confidence"][i]*100:0.2f}%'
        cv2.putText(result, text, text_start, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1,  cv2.LINE_AA)
            
        # Draw bouding box with the above measurements
        cv2.rectangle(result, box_start, box_end, (0,255,0), 1)
            

    # Display text about confidence rate above each box
    text = f'Number of faces detected: {len(global_cache["topleft"])}'    
    cv2.putText(result, text,(10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0), 1, cv2.LINE_AA)
    
    # Display the frame
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