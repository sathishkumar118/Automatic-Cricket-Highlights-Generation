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
from moviepy.tools import subprocess_call
from moviepy.config import get_setting
import numpy as np

#from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

match_name="3_BBL_Trim"
source="C:/FYP/Videos/"+match_name+".mp4"

#"C:/FYP/Videos/6_BBL_Trim.mp4"
end = time.time()
dur = end-start
if dur<60:
    print("import Time:",dur,"seconds")
elif dur>60 and dur<3600:
    dur=dur/60
    print("import Time:",dur,"minutes")
else:
    dur=dur/(60*60)
    print("import Time:",dur,"hours")

#Define path for inary classifiers	
#Define Path
start = time.time()
#Load the pre-trained models
model1 = load_model("./OVA_models/1_commentators_model.h5")
model1.load_weights("./OVA_models/1_commentators_weights.h5")

model2 = load_model('./OVA_models/2_interviews_model.h5')
model2.load_weights('./OVA_models/2_interviews_weights.h5')

model3 = load_model('./OVA_models/3_crowd_model.h5')
model3.load_weights('./OVA_models/3_crowd_weights.h5')

model4 = load_model('./OVA_models/4_field_view_model.h5')
model4.load_weights('./OVA_models/4_field_view_weights.h5')

model5 = load_model('./OVA_models/5_players_gathering_model.h5')
model5.load_weights('./OVA_models/5_players_gathering_weights.h5')


model6 = load_model('./OVA_models/6_pitch_view_model.h5')
model6.load_weights('./OVA_models/6_pitch_view_weights.h5')

model7= load_model('./OVA_models/7_pitch_view_model.h5')
model7.load_weights('./OVA_models/7_pitch_view_weights.h5')

model8= load_model('./OVA_models/8_closeup-1_model.h5')
model8.load_weights('./OVA_models/8_closeup-1_weights.h5')

model9= load_model('./OVA_models/9_longshot_model.h5')
model9.load_weights('./OVA_models/9_longshot_weights.h5')

model10= load_model('./OVA_models/10_closeup-2_model.h5')
model10.load_weights('./OVA_models/10_closeup-2_weights.h5')

model11= load_model('./OVA_models/11_closeup-3_model.h5')
model11.load_weights('./OVA_models/11_closeup-3_weights.h5')

model12= load_model('./OVA_models/12_sideview_model.h5')
model12.load_weights('./OVA_models/12_sideview_weights.h5')


end = time.time()
dur = end-start
if dur<60:
    print("load Time:",dur,"seconds")
elif dur>60 and dur<3600:
    dur=dur/60
    print("load Time:",dur,"minutes")
else:
    dur=dur/(60*60)
    print("load Time:",dur,"hours")
	
start = time.time()	
with open("C:/FYP/output/"+match_name+'_FrameAnnotation.csv', mode='w') as csv_file:
	field = ['time', 'start of the ball','single_player','commentators', 'interviews', 'crowd','field_view','playergathering','others']
	writer1 = csv.DictWriter(csv_file, fieldnames=field)
	writer1.writeheader()

"""def multi_predict(x):
	test_image = cv2.resize(x,(150,150))
	test_image = img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis=0)
	array = multimodel.predict(test_image)
	result = array[0]
	print(result)
	answer = np.argmax(result)
	print(answer)
	time=str(i)+"s"
	zero="0"
	one="0"
	two="0"
	three="0"
	four="0"
	five="0"
	six="0"
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
	myfield=[time,zero,'0',one,two,three,four,five,six]
	with open("C:/FYP/output/"+match_name+'_FrameAnnotation.csv','a') as cs_file:
		#field = ['time', 'start of the ball', 'commentators', 'interviews', 'field_view','crowd','playergathering','others']
		writer1 = csv.writer(cs_file)
		writer1.writerow(myfield)"""
	
def models_predict(x,i):
	test_image = cv2.resize(x,(64,64))
	test_image = img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis = 0)
	time=str(i)+"s"
	zero=one=two=three=four=five=six=seven="0"
	ball[i]=0

	if model8.predict(test_image)[0][0]==0:
		one="1"
	elif model10.predict(test_image)[0][0]==0:
		one="1"
	elif model11.predict(test_image)[0][0]==0:
		one="1"
	elif model3.predict(test_image)[0][0]==0:
		four="1"
	elif model4.predict(test_image)[0][0]==0:
		five="1"
	elif model5.predict(test_image)[0][0]==0:
		six="1"
	elif model9.predict(test_image)[0][0]==0:
		seven="1"
	elif model12.predict(test_image)[0][0]==0:
		seven="1"
	elif model1.predict(test_image)[0][0]==0:
		two="1"
	elif model2.predict(test_image)[0][0]==0:
		three="1"
	elif model6.predict(test_image)[0][0]==0:
		zero="1"
		ball[i]=1
	elif model7.predict(test_image)[0][0]==0:
		zero="1"
		ball[i]=1
	
#record the results in a CSV file

	with open("C:/FYP/output/"+match_name+'_FrameAnnotation.csv','a') as cs_file:
		#field = ['time', 'start of the ball', 'single player',commentators', 'interviews', ,'crowd','field_view','playergathering','others']
		writer1 = csv.writer(cs_file)
		writer1.writerow([str(str(i)+'s'),zero,one,two,three,four,five,six,seven])



	"""if i>=60:
		print(str(i)+" "+str(int((i-(i%60))/60))+":"+str(i%60)+" scorecard detected")
		print("start of the ball!!!\n")
		ball[i]=1
	else:
		print(str(i)+"s scorecard detected\nstart of the ball!!!\n")
		ball[i]=1
def single_predict(x,i):
	x1 = cv2.resize(x,(150,150))
	x1 = img_to_array(x1)
	x1 = np.expand_dims(x1, axis=0)
	array = single_model.predict(x1)
	result = array[0]
	#print(result)

	answer = np.argmax(result)
	if answer <5:
		with open("C:/FYP/output/"+match_name+'_FrameAnnotation.csv','a') as cs_file:
				#field = ['time', 'start of the ball','single_player','commentators', 'interviews', 'field_view','crowd','playergathering','others']
			writer1 = csv.writer(cs_file)
			writer1.writerow([str(str(i)+'s'),'0',str(answer),'0','0','0','0','0','0'])
			ball[i]=0
	elif answer == 5:
		mypredict(x,i)
	
def single_predict(x,i):
	test_image = cv2.resize(x,(64,64))
	test_image = img_to_array(test_image)
	test_image = np.expand_dims(test_image, axis = 0)
	result=single_model.predict(test_image)
	#print(result)
	array=result[0][0]
	#answer = np.argmax(array)
	if array == 0:
		with open('FrameAnnotation.csv','a') as cs_file:
				#field = ['time', 'start of the ball','single_player','commentators', 'interviews', 'field_view','crowd','playergathering','others']
			writer1 = csv.writer(cs_file)
			writer1.writerow([str(str(i)+'s'),'0','1','0','0','0','0','0','0'])
		if i>=60:
			print(str(i)+" "+str(int((i-(i%60))/60))+":"+str(i%60)+" scorecard detected")
			print("start of the ball!!!\n")
			#ball[i]=1
		else:
			print(str(i)+"s scorecard detected\nstart of the ball!!!\n")
			#ball[i]=1
	else:
		#
		print("other frame\n")
		mypredict(x,i)
"""		
		
		
video = VideoFileClip(source)
print(video.duration)
duration=int(video.duration)
i=0
n=1
ball=[-1]*int(duration)
a=i
b=i
flag=2
score=[0]*(duration+1)
wicket=[0]*(duration+1)
replay=[]
last=""
last_time=0;
with open("C:/FYP/output/"+match_name+'_preprocessing.csv', mode='w') as csv_file:
	fieldnames = ['time', 'scorecard present', 'scorecard absent']
	writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
	writer.writeheader()
    #writer.writerow({'emp_name': 'John Smith', 'dept': 'Accounting', 'birth_month': 'November'})
    #writer.writerow({'emp_name': 'Erica Meyers', 'dept': 'IT', 'birth_month': 'March'})


	while i<duration:
		vidcap = cv2.VideoCapture(source)
		vidcap.set(cv2.CAP_PROP_POS_MSEC,i*1000)
		success,myimg=vidcap.read()
		image=myimg
		myimg = cv2.cvtColor(myimg, cv2.COLOR_BGR2GRAY)
		new_image=myimg
		new_image = cv2.convertScaleAbs(myimg, alpha=0.90, beta=50)
		#myimg = cv2.convertScaleAbs(myimg, alpha=2.0, beta=0)
		#kernel = np.ones((1, 1), np.uint8)
		#myimg = cv2.dilate(myimg, kernel, iterations=1)
		#myimg = cv2.erode(myimg, kernel, iterations=1)
		#[up:down,left:right]
		cropped=new_image[630:660, 266:346]
		result = pytesseract.image_to_string(cropped)
		result=result.replace(" ","")
		#result=float(result)
		if len(result)>0:
			if re.match(r'^[0-9]*/[0-9]*$',result):
				br=result.find('/')
				print(str(i)+"/"+str(duration),end="\r")
				#(result)
				last=result
				last_time=i;
				wicket[i]=int(result[:br])
				score[i]=int(result[br+1:])
				#print(str(i)+"th second\nscorecard detected\n")
				if flag==2:
					a=i
					b=i
					flag=1
				elif flag==0:
					replay.append(a)
					if a<b:
						writer.writerow({'time': str(a)+'s-'+str(b)+'s','scorecard present':'0', 'scorecard absent':'1'})
						print(str(a)+"s - "+str(b)+"s\t-> Scorecard absent")
					else:
						writer.writerow({'time': str(a)+"s",'scorecard present':'0', 'scorecard absent':'1'})
						print(str(a)+"s - "+str(b)+"s\t-> Scorecard absent")
					a=i
					b=i
					flag=1
				elif flag==1:
					b+=1
				ball[i]=0
				models_predict(image,i)
			else:
				#print(str(i)+"th second\nscorecard not detected\n")
				if flag==2:
					a=i
					b=i
					flag=0
				elif flag==1:
					if a<b:
						writer.writerow({'time': str(a)+'s-'+str(b)+'s','scorecard present':'1', 'scorecard absent':'0'})
						print(str(a)+"s - "+str(b)+"s\t-> Scorecard present")
					else:
						writer.writerow({'time': str(a)+"s",'scorecard present':'1', 'scorecard absent':'0'})
					a=i
					b=i
					flag=0
				elif flag==0:
					b+=1
				ball[i]=-1
		else:
			cropped1=new_image[650:685, 275:340]
			result = pytesseract.image_to_string(cropped1)
			if re.match(r'[0-9]*/[0-9]*',result):
				br=result.find('/')
				print(str(i)+"/"+str(duration),end="\r")
				#print(result)
				last=result
				last_time=i
				wicket[i]=int(result[:br])
				score[i]=int(result[br+1:])
				#print(str(i)+"th second\nscorecard detected\n")
				if flag==2:
					a=i
					b=i
					flag=1
				elif flag==0:
					replay.append(a)
					if a<b:
						writer.writerow({'time': str(a)+'s-'+str(b)+'s','scorecard present':'0', 'scorecard absent':'1'})
						print(str(a)+"s - "+str(b)+"s\t-> Scorecard absent")
					else:
						writer.writerow({'time': str(a)+"s",'scorecard present':'0', 'scorecard absent':'1'})
					a=i
					b=i
					flag=1
				elif flag==1:
					b+=1
				ball[i]=0
				models_predict(image,i)
			else:
				#print(str(i)+"th second\nscorecard not detected\n")
				if flag==2:
					a=i
					b=i
					flag=0
				elif flag==1:
					if a<b:
						writer.writerow({'time': str(str(a)+"s-"+str(b)+'s'),'scorecard present':'1', 'scorecard absent':'0'})
						print(str(a)+"s - "+str(b)+"s\t-> Scorecard present")
					else:
						writer.writerow({'time': str(a)+"s",'scorecard present':'1', 'scorecard absent':'0'})
					a=i
					b=i
					flag=0
				elif flag==0:
					b+=1
				ball[i]=-1
		i+=1
	if flag==1:
		writer.writerow({'time': str(a)+"s-"+str(b)+'s','scorecard present':'1', 'scorecard absent':'0'})
		print(str(a)+"s- "+str(b)+"s\t-> Scorecard present")
	elif flag==0:
		writer.writerow({'time': str(a)+"s-"+str(b)+'s','scorecard present':'0', 'scorecard absent':'1'})
		print(str(a)+"s - "+str(b)+"s\t-> Scorecard absent")


def ffmpeg_extract_subclip(filename, t1, t2, targetname=None):
    """ makes a new video file playing video file ``filename`` between
        the times ``t1`` and ``t2``. """
    name,ext = os.path.splitext(filename)
    if not targetname:
        T1, T2 = [int(1000*t) for t in [t1, t2]]
        targetname = name+ "%sSUB%d_%d.%s"(name, T1, T2, ext)
    
    cmd = [get_setting("FFMPEG_BINARY"),"-y",
      "-ss", "%0.2f"%t1,"-i", filename,
      "-t", "%0.2f"%(t2-t1),
      "-vcodec", "copy", "-acodec", "copy", targetname]


	
def checknext(it,ball):
	ref=ball[it:it+11]
	if -1 not in ref:
		if sum(ball[it:it+11]) >1:
			return True
		else:
			return False
	else:
		return False

score=score[2:]
wicket=wicket[2:]
score.append(score[-1])
score.append(score[-1])
wicket.append(wicket[-1])
wicket.append(wicket[-1])
print("finalyy!!\n\n\n")
if len(last) is 0:
	print("There is no live frames in the video")
else:
	with open("C:/FYP/output/"+match_name+'_BallBoundary.csv', mode='w') as bb_file:
		field = ['count', 'start of the ball','end of the ball','score']
		writer1 = csv.DictWriter(bb_file, fieldnames=field)
		writer1.writeheader()
	with open("C:/FYP/output/"+match_name+'_ScorecardHighlights.csv', mode='w') as sr_file:
		field = ['count', 'start of the ball','end of the ball','type']
		writer1 = csv.DictWriter(sr_file, fieldnames=field)
		writer1.writeheader()
	it=0	
	result=[]
	while it<len(ball):
		if ball[it]==1:
			print(str(it)+"th frame")
			if checknext(it,ball):
				result.append(it)
				it+=12
		it+=1
	print(result)
	it=1
	sh=0
	hs_start=[]
	hs_end=[]

	result.append(duration)
	wicket[result[-1]]=int(last[:br])
	score[result[-1]]=int(last[br+1:])
	print(result)
	j=0
	print(replay)
	hcount=1
	while sh<len(result):
		with open("C:/FYP/output/"+match_name+'_BallBoundary.csv','a') as bb_file:
			#field = ['time', 'start of the ball','end of the ball','score']
			writer1 = csv.writer(bb_file)
			if(sh+1<len(result)):
				writer1.writerow([str(it),str(result[sh]),str(result[sh+1]),str(score[result[sh]])+"/"+str(wicket[result[sh]])])
		print("score at "+str(it)+"th delivery "+str(result[sh])+"th frame is "+str(score[result[sh]])+"/"+str(wicket[result[sh]]))
		if sh+1<len(result) and score[result[sh+1]]-score[result[sh]]>=4 :
			print("highlights 4s/6s \n")
			hs_start.append(result[sh])
			while(j<len(replay) and hs_start[-1]>replay[j]):
				j+=1
			if(j<len(replay) and replay[j]<result[sh+1]):
				hs_end.append(replay[j])
			else:
				hs_end.append(result[sh+1])
			with open("C:/FYP/output/"+match_name+'_ScorecardHighlights.csv','a') as sr_file:
				#field = ['time', 'start of the ball','end of the ball','type']
				writer1 = csv.writer(sr_file)
				writer1.writerow([str(hcount),str(hs_start[-1]),str(hs_end[-1]),"Boundary"])
			hcount+=1
		elif sh+1<len(result) and wicket[result[sh+1]]-wicket[result[sh]]>=1 : 
			print("highlights wicket \n")
			hs_start.append(result[sh])
			while(j<len(replay) and hs_start[-1]>replay[j]):
				j+=1
			if(j<len(replay) and replay[j]<result[sh+1]):
				hs_end.append(replay[j])
			else:
				hs_end.append(result[sh+1])
			with open("C:/FYP/output/"+match_name+'_ScorecardHighlights.csv','a') as sr_file:
				#field = ['time', 'start of the ball','end of the ball','type']
				writer1 = csv.writer(sr_file)
				writer1.writerow([str(hcount),str(hs_start[-1]),str(hs_end[-1]),"Wicket"])
			hcount+=1
		it+=1
		sh+=1
	#hs_start.append(int(duration-1))

	print(hs_start)
	print(hs_end)
	#hs_start=[1, 17, 33, 50, 68, 89]
	#hs_end=[17, 33, 50, 68, 89, 108]
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
	#aggregation
	#video= mp.VideoFileClip("C:/FYP/Videos/BBL_Trim.mp4")
	clips= []
	i=0
	while i<len(hs_start):
	#create subclip for given time in secs
		"""target="out"+str(i)
		ffmpeg_extract_subclip(source, hs_start[i], hs_end[i], targetname="C:/FYP/code/"+target+".mp4")"""
		if hs_start[i]-2>0:
			clip = video.subclip(hs_start[i]-1,hs_end[i]-1)
			clips.append(clip)
		else:
			clip = video.subclip(hs_start[i],hs_end[i]-1)
			clips.append(clip)
		#source="C:/FYP/code/"+target+".mp4"
		i+=1
	#concatenating shots
	
	if len(clips)>0:
		final_clip = mp.concatenate_videoclips(clips)
		final_clip.write_videofile(source.replace(".mp4","_highlights.mp4"), codec='libx264',fps=15,audio_bitrate="30k",preset="ultrafast",threads=64)
	else:
		print("There is no live frames in the video")

#--------	
	
end = time.time()
dur = end-start
if dur<60:
    print("aggregation Time:",dur,"seconds")
elif dur>60 and dur<3600:
    dur=dur/60
    print("aggregation Time:",dur,"minutes")
else:
    dur=dur/(60*60)
    print("aggregation Time:",dur,"hours")

