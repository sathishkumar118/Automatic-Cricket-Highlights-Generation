import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from keras.models import Sequential, load_model
import time

start = time.time()

#Define Path
model_path = './six_class_models/model.h5'
model_weights_path = './six_class_models/weights.h5'
test_path = './testing'

#Load the pre-trained models
model = load_model(model_path)
model.load_weights(model_weights_path)

#Define image parameters
img_width, img_height = 150, 150

#Prediction Function
def predict(file):
	x = load_img(file, target_size=(img_width,img_height))
	x = img_to_array(x)
	x = np.expand_dims(x, axis=0)
	array = model.predict(x)
	result = array[0]
	#print(result)
	answer = np.argmax(result)
	if answer == 0:
		six="1"
		print("Predicted: other")
	elif answer == 1:
		one="1"
		print("Predicted: commentators")
	elif answer == 2:
		two="1"
		print("Predicted: interviews")
	elif answer == 3:
		three="1"
		print("Predicted: crowd")
	elif answer == 4:
		four="1"
		print("Predicted: field_view")
	elif answer == 5:
		five="1"
		print("Predicted: players gathering")
	return answer

#Walk the directory for every image
for i, ret in enumerate(os.walk(test_path)):
  for i, filename in enumerate(ret[2]):
    if filename.startswith("."):
      continue
    
    print(ret[0] + '/' + filename)
    result = predict(ret[0] + '/' + filename)
    print(result)

#Calculate execution time
end = time.time()
dur = end-start

if dur<60:
    print("Execution Time:",dur,"seconds")
elif dur>60 and dur<3600:
    dur=dur/60
    print("Execution Time:",dur,"minutes")
else:
    dur=dur/(60*60)
    print("Execution Time:",dur,"hours")
