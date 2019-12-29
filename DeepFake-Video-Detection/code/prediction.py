import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import model_from_json

#load model from json file
json_file = open('models/model_2.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("models/model_2.h5")

# Compile model
loaded_model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

trfv = 0
#prediction_for_real
for i in range(1,50):
    video_str = "fake_videos/real/_ ("+str(i)+")"
    cap = cv2.VideoCapture(video_str)

    count = 0
    tf_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            img = cv2.resize(frame,(224,224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis = 0)
            result = loaded_model.predict(img)
            
            if result[0][0]==1:
                tf_count += 1
            if tf_count>=5:
                print(video_str+"is fake")
                trfv += 1
                break
            count += 1
            cap.set(1, count)
        else:
            cap.release()
            break

    if tf_count<5:
        print(video_str+"is real")

tfrv = 0
#prediction_for_fake
for i in range(1,50):
    video_str = "fake_videos/fake/_ ("+str(i)+")"
    cap = cv2.VideoCapture(video_str)

    count = 0
    tf_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            img = cv2.resize(frame,(224,224))
            img = image.img_to_array(img)
            img = np.expand_dims(img, axis = 0)
            result = loaded_model.predict(img)
            
            if result[0][0]==1:
                tf_count += 1
            if tf_count>=5:
                print(video_str+"is fake")
                tfrv += 1
                break
            count += 1
            cap.set(1, count)
        else:
            cap.release()
            break
        
    if tf_count<5:
        print(video_str+"is real")

accuracy = ((48-trfv)+tfrv)/98
print(accuracy)
