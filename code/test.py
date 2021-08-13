import time
start = time.time()
import moviepy.editor as mp
import cv2
import numpy as np
import pytesseract
import re
from PIL import Image
from moviepy.editor import VideoFileClip
import os
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
from keras.preprocessing import image
import time
import csv
print("import complete")
start = time.time()
vidcap = cv2.VideoCapture("C:/FYP/Videos/1_BBL_Trim_Highlights.mp4")
vidcap.set(cv2.CAP_PROP_POS_MSEC,2*1000)
success,myimg=vidcap.read()
image=myimg
test_image = cv2.resize(image,(64,64))
test_image = img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
end = time.time()
dur = end-start
if dur<60:
    print("image frame Time:",dur,"seconds")
elif dur>60 and dur<3600:
    dur=dur/60
    print("image frame Time:",dur,"minutes")
else:
    dur=dur/(60*60)
    print("image frame Time:",dur,"hours")
	
start = time.time()
cumulated_model_path = './models/cnn_models/model.h5'
cumulated_model_weights_path = './models/cnn_models/model.h5'
cumulated_model = load_model(cumulated_model_path)
cumulated_model.load_weights(cumulated_model_weights_path)
#array = cumulated_model.predict(test_image)
#result = array[0]
#print(result)


cumulated_model1 = load_model(cumulated_model_path)
cumulated_model1.load_weights(cumulated_model_weights_path)

cumulated_model2 = load_model(cumulated_model_path)
cumulated_model2.load_weights(cumulated_model_weights_path)

cumulated_model3 = load_model(cumulated_model_path)
cumulated_model3.load_weights(cumulated_model_weights_path)

cumulated_model4 = load_model(cumulated_model_path)
cumulated_model4.load_weights(cumulated_model_weights_path)

cumulated_model5 = load_model(cumulated_model_path)
cumulated_model5.load_weights(cumulated_model_weights_path)


cumulated_model6= load_model(cumulated_model_path)
cumulated_model6.load_weights(cumulated_model_weights_path)

cumulated_model7= load_model(cumulated_model_path)
cumulated_model7.load_weights(cumulated_model_weights_path)

cumulated_model8= load_model(cumulated_model_path)
cumulated_model8.load_weights(cumulated_model_weights_path)

cumulated_model9= load_model(cumulated_model_path)
cumulated_model9.load_weights(cumulated_model_weights_path)


cumulated_model10= load_model(cumulated_model_path)
cumulated_model10.load_weights(cumulated_model_weights_path)

cumulated_model11= load_model(cumulated_model_path)
cumulated_model11.load_weights(cumulated_model_weights_path)

cumulated_model12= load_model(cumulated_model_path)
cumulated_model12.load_weights(cumulated_model_weights_path)


end = time.time()
dur = end-start
if dur<60:
    print("execution Time:",dur,"seconds")
elif dur>60 and dur<3600:
    dur=dur/60
    print("execution Time:",dur,"minutes")
else:
    dur=dur/(60*60)
    print("execution Time:",dur,"hours")

	
start = time.time()
i=0
while i<480:
	i+=1
	array = cumulated_model.predict(test_image)
	result = array[0]
	print(result)

	array = cumulated_model1.predict(test_image)
	result = array[0]
	print(result)

	array = cumulated_model2.predict(test_image)
	result = array[0]
	print(result)

	array = cumulated_model3.predict(test_image)
	result = array[0]
	print(result)

	array = cumulated_model4.predict(test_image)
	result = array[0]
	print(result)

	array = cumulated_model5.predict(test_image)
	result = array[0]
	print(result)

	array = cumulated_model6.predict(test_image)
	result = array[0]
	print(result)

	array = cumulated_model7.predict(test_image)
	result = array[0]
	print(result)

	array = cumulated_model8.predict(test_image)
	result = array[0]
	print(result)

	array = cumulated_model9.predict(test_image)
	result = array[0]
	print(result)

	array = cumulated_model10.predict(test_image)
	result = array[0]
	print(result)

	array = cumulated_model11.predict(test_image)
	result = array[0]
	print(result)

	array = cumulated_model12.predict(test_image)
	result = array[0]
	print(result)


end = time.time()
dur = end-start
if dur<60:
    print("prediction Time:",dur,"seconds")
elif dur>60 and dur<3600:
    dur=dur/60
    print("prediction Time:",dur,"minutes")
else:
    dur=dur/(60*60)
    print("prediction Time:",dur,"hours")