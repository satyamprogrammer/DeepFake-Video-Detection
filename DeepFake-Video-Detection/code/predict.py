import cv2
import shutil
import os
import argparse
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json

if __name__ == '__main__':
    parser = parser = argparse.ArgumentParser(description='VideoDetect')
    parser.add_argument('--vid', required=True, help='Path for source video')
    args = parser.parse_args()
    
    #load model from json file
    json_file = open('models/model_0.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("models/model_0.h5")

    # Compile model
    loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

    #Opens the Video file
    cap = cv2.VideoCapture(args.vid)
    fake_count = 0
    i=0
    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        img = cv2.resize(frame,(299,299))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis = 0)
        result = loaded_model.predict(img)
        if result[0][0] == 0:
            fake_count += 1
        i+=1
    
    cap.release()
    cv2.destroyAllWindows()

    print('Total fake frames counted = '+str(fake_count))
    print('Total frames = '+str(i), end = '\n\n')
    print('Percentage of fake frames counted = '+str((fake_count/i)*100))
