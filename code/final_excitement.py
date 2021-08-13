import numpy as np # for numerical operations
from moviepy.editor import VideoFileClip
clip=VideoFileClip("C:/FYP/Videos/6_BBL_Trim.mp4")
duration=clip.audio.duration
j=0
e=[]
V=2200
while j<duration:
	#print(j)
	sarr=clip.audio.subclip(j,j+1).to_soundarray(fps=V)
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
i=0
while i<5:
	ae.append(np.mean((e[j+i:])))
	i=i+1
print("\n\naverage")
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

#new_block

j=0
psi=[]
while j<len(ne):
	psi.append(int(ne[j]>=p_audio))
	j+=1
print("psi")
start=[15,62,96,129,166,236,271,309]
end=[31,96,129,145,183,254,292,329]
j=0
print(psi)
while j<len(start):
	print(sum(psi[start[j]:end[j]])/len(psi[start[j]:end[j]]))
	j=j+1
#print(sum(psi[4:24])/len(psi[4:24]))
