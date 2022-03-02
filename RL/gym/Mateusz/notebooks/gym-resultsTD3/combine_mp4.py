from moviepy.editor import VideoFileClip, concatenate_videoclips
import os

# grab all mp4 files in the current directory
mp4_files = [f for f in os.listdir() if f.endswith('.mp4')]


# Concatenate all mp4 files into one video

for i in range(len(mp4_files)):

    mp4_files[i] = os.path.join(os.getcwd(), mp4_files[i])


# make all videos clip objects 
clips = [VideoFileClip(mp4_file) for mp4_file in mp4_files]


final_clip = concatenate_videoclips(clips)

final_clip.write_videofile('combined_video.mp4')