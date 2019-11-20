
# coding: utf-8
# Final runnable code which takes the full match video as input and gives as highlights as output
# In[2]:


import time
start = time.time()
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
#from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip



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
	


# In[20]:
# importing moviepy editor to concatenate the video clips
import moviepy.editor as mp

match_name="3_BBL_Trim"
source="C:/FYP/Videos/"+match_name+".mp4"
myclip=VideoFileClip(source)
duration=myclip.audio.duration
j=0
e=[]
V=2200
while j<duration:
	#print(j)
	sarr=myclip.audio.subclip(j,j+1).to_soundarray(fps=V)
	i=j*V+1
	#while i<=(j+1)*2200:
	e.append(((1.0*sarr)**2).mean())
	j+=1
print(e)
j=0
L=5
ae=[]
while j<duration-L:
	ae.append(np.mean((e[j:j+L])))
	j+=1
print("average")
print(ae)
ne=[]
j=0
while j<len(ae):
	ne.append(ae[j]/max(ae))
	j+=1
print("normalised energy")
print(ne)
print("p_audio")
p_audio=np.mean(ne)
print(p_audio)
j=0
psi=[]
while j<len(ne):
	psi.append(int(ne[j]>=p_audio))
	j+=1
print("psi")
print(psi)

def excitement(start,end):
	prob=[]
	hs_ex=[]
	j=0
	while j<len(start):
		prob.append(sum(psi[start[j]:end[j]])/len(psi[start[j]:end[j]]))
		print(prob[-1])
		j=j+1
	thres=np.mean(prob)+np.var(prob)
	i=0
	print(prob)
	while i<len(prob):
		if prob[i]>=thres:
			hs_ex.append(i)
		i+=1
	#print(hs_ex)
	return hs_ex


start = time.time()	
with open("C:/FYP/output/"+match_name+'_FrameAnnotation.csv', mode='w') as csv_file:
	field = ['time', 'start of the ball','single_player','commentators', 'interviews', 'crowd','field_view','playergathering','others']
	writer1 = csv.DictWriter(csv_file, fieldnames=field)
	writer1.writeheader()


# In[18]:


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



start = time.time()
video = myclip
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
	while i<duration:
		vidcap = cv2.VideoCapture(source)
		vidcap.set(cv2.CAP_PROP_POS_MSEC,i*1000)
		success,myimg=vidcap.read()
		image=myimg
		myimg = cv2.cvtColor(myimg, cv2.COLOR_BGR2GRAY)
		new_image=myimg
		new_image = cv2.convertScaleAbs(myimg, alpha=0.90, beta=50)
		cropped=new_image[630:660, 266:346]
		result = pytesseract.image_to_string(cropped)
		result=result.replace(" ","")
		#result=float(result)
		if len(result)>0:
			if re.match(r'^[0-9]*/[0-9]*$',result):
				br=result.find('/')
				print(str(i)+"/"+str(duration),end="\r")
				#print(result)
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
						print(str(a)+"s\t-> Scorecard absent")
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
def checknext(it,ball):
	ref=ball[it:it+11]
	if -1 not in ref:
		if sum(ball[it:it+11]) >=1:
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
print("done...\n")
def score_recog(start,end):
    i=0
    hs=[]
    while(i<len(start)):
        if i+1<len(start) and score[start[i+1]]-score[start[i]]>=4:
            hs.append(i)
            with open("C:/FYP/output/"+match_name+'_ScorecardHighlights.csv','a') as sr_file:
                    #field = ['time', 'start of the ball','end of the ball','type']
                    writer1 = csv.writer(sr_file)
                    writer1.writerow([str(len(hs)),str(start[i]),str(end[i]),"Boundary"])
        elif i+1<len(start) and wicket[start[i+1]]-wicket[start[i]]>=1:
            hs.append(i)
            with open("C:/FYP/output/"+match_name+'_ScorecardHighlights.csv','a') as sr_file:
                    #field = ['time', 'start of the ball','end of the ball','type']
                    writer1 = csv.writer(sr_file)
                    writer1.writerow([str(len(hs)),str(start[i]),str(end[i]),"Wicket"])
        elif score[end[i]]-score[start[i]]>=4:
            hs.append(i)
            with open("C:/FYP/output/"+match_name+'_ScorecardHighlights.csv','a') as sr_file:
                    #field = ['time', 'start of the ball','end of the ball','type']
                    writer1 = csv.writer(sr_file)
                    writer1.writerow([str(len(hs)),str(start[i]),str(end[i]),"Boundary"])
        elif wicket[end[i]]-wicket[start[i]]>=1:
            hs.append(i)
            with open("C:/FYP/output/"+match_name+'_ScorecardHighlights.csv','a') as sr_file:
                    #field = ['time', 'start of the ball','end of the ball','type']
                    writer1 = csv.writer(sr_file)
                    writer1.writerow([str(len(hs)),str(start[i]),str(end[i]),"Wicket"])
        i+=1
    return hs
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
			#print(str(it)+"th frame")
			if checknext(it,ball):
				result.append(it)
				it+=12
		it+=1
	print(result)
	it=1
	sh=0
	hs_start=[]
	hs_end=[]
	if(replay[-1]>result[-1]):
		result.append(replay[-1])
	else:
		result.append(duration)
	wicket[result[-1]]=int(last[:br])
	score[result[-1]]=int(last[br+1:])
	print(result)
	j=0
	print(replay)
	hcount=1
	i=0
	b_start=result[:-1]
	b_end=[]
	while i+1<len(result):
		j=0
		while j<len(replay)and result[i]>replay[j]:
			j+=1
		if(j<len(replay) and replay[j]<result[i+1]):
			b_end.append(replay[j])
		else:
			b_end.append(result[i+1])
		i+=1
	j=0
	print(b_start)
	print(b_end)
	while sh<len(result):
		with open("C:/FYP/output/"+match_name+'_BallBoundary.csv','a') as bb_file:
			#field = ['time', 'start of the ball','end of the ball','score']
			writer1 = csv.writer(bb_file)
			if(sh+1<len(result)):
				writer1.writerow([str(it),str(b_start[sh]),str(b_end[sh]),str(score[result[sh]])+"/"+str(wicket[result[sh]])])
		print("score at "+str(it)+"th delivery "+str(result[sh])+"th frame is "+str(score[result[sh]])+"/"+str(wicket[result[sh]]))
		sh+=1
	he=excitement(b_start,b_end)
	hs=score_recog(b_start,b_end)
	#hs_start.append(int(duration-1))
	print(hs)
	print(he)
	#hs_start=[1, 17, 33, 50, 68, 89]
	#hs_end=[17, 33, 50, 68, 89, 108]


hs_start=[]
hs_end=[]
i=0
while i<len(b_start):
    if i in hs or i in he:
        hs_start.append(b_start[i])
        hs_end.append(b_end[i])
    i+=1
print(hs_start)
print(hs_end)
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


# In[19]:


start = time.time()
#aggregation
#video= mp.VideoFileClip("C:/FYP/Videos/BBL_Trim.mp4")
clips= []
i=0
while i<len(hs_start):
#create subclip for given time in secs
	"""target="out"+str(i)
	ffmpeg_extract_subclip(source, hs_start[i], hs_end[i], targetname="C:/FYP/code/"+target+".mp4")"""
	if hs_start[i]+1<duration and hs_end[i]-4>0:
		clip = video.subclip(hs_start[i]+1,hs_end[i]-4)
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

