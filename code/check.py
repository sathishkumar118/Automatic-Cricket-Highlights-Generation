import subprocess
ffmpeg_command1 = ["ffmpeg", "-i", "test_1.mp4", "-acodec", "copy", "-ss", "00:00:00", "-t", "00:00:30", "test_1.mp4"]
ffmpeg_command2 = ["ffmpeg", "-i","test_2.mp4", "-acodec", "copy", "-ss", "00:00:30", "-t", "00:00:30", "test_2.mp4"]
ffmpeg_command3 = ["mencoder", "-forceidx", "-ovc", "copy", "-oac", "pcm", "-o", "output.mp4", "test_1.mp4", "test_2.mp4"]


#subprocess.call(ffmpeg_command1)
#subprocess.call(ffmpeg_command2)
subprocess.Popen(ffmpeg_command3)