import moviepy.editor as mp
import cv2
import os
import time
"""start=[25,85]
end=[39,90]
video1= mp.VideoFileClip("test_1.mp4")
video2= mp.VideoFileClip("test_2.mp4")
clips= []
i=0
#while i<len(start) and i<len(end):
#create subclip for given time in secs
clip1=video1.subclip(0,100)
clip2=video2.subclip(0,100)
"""
"""if start[i]<end[i]:
	clip = video.subclip(0,10)
	clips.append(clip)
i+=1"""
"""
#concatenating shots
final_clip = mp.concatenate_videoclips([clip1,clip2])
#cv2.imwrite("my_concatenation.mp4",final_clip)
final_clip.write_videofile("output.mp4")
"""
hs_start=[1, 17]
hs_end=[17, 33]
#hs_start=[1, 17, 33, 50, 68, 89]
#hs_end=[17, 33, 50, 68, 89, 108]
start = time.time()	
#aggregation
video= mp.VideoFileClip("C:/FYP/Videos/5_BBL_Trim.mp4")
clips= []
i=0
while i<len(hs_start):
#create subclip for given time in secs
	"""target="out"+str(i)
	ffmpeg_extract_subclip(source, hs_start[i], hs_end[i], targetname="C:/FYP/code/"+target+".mp4")"""
	if hs_start[i]-2>0:
		clip = video.subclip(hs_start[i],hs_end[i]-1)
		clips.append(clip)
	else:
		clip = video.subclip(hs_start[i],hs_end[i]-1)
		clips.append(clip)
	#source="C:/FYP/code/"+target+".mp4"
	i+=1
#concatenating shots

if len(clips)>0:
	final_clip = mp.concatenate_videoclips(clips)
	final_clip.write_videofile("C:/FYP/Videos/5_BBL_Trim__test.mp4", codec='libx264',fps=15,audio_bitrate="30k",preset="ultrafast",threads=64)
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
