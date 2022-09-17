# -*- coding: utf-8 -*-
"""
Created on Fri Dec  4 20:47:06 2020

@author: Student
"""

import youtube_dl
import os,subprocess
import random

video_link = 'https://www.youtube.com/watch?v=XFH6IQk4YZQ'
video_height = 240
video_width = 426
video_res = '240p'
video_format = 'mp4'
video_fps = 30

min_video_duration = 50  # in seconds
def get_url_to_download(video_link):
    options = {
        'format': 'bestaudio/best',  # choice of quality
        'extractaudio': True,        # only keep the audio
        'audioformat': "mp3",        # convert to mp3
        'outtmpl': '%(id)s',         # name the file the ID of the video
        'noplaylist': True,          # only download single song, not playlist
        #'listformats': True,         # print a list of the formats to stdout and exit
    }
    ydl = youtube_dl.YoutubeDL(options)
    with ydl:
        try:
            result = ydl.extract_info(
                video_link,
                download=False # We just want to extract the info
            )
        except :
            return -1,-1
    if 'entries' in result:
        # Can be a playlist or a list of videos
        video = result['entries'][0]
    else:
        # Just a video
        video = result
    json = video['formats']
    # extracting only video part since audio part is needed
    # audio can also be extracted in same format , please check video dictonary
    url = -1
    for i in range(len(json)):
       if json[i]['ext'] == video_format and json[i]['height'] == video_height and json[i]['format_note'] == video_res and json[i]['width'] == video_width and json[i]['fps'] == video_fps:
           url = json[i]['url']
           break

    return url,video['duration']

def download_video(url,start_time,video_name):
    #os.close()
    #os.remove("D:/python/New/Results/testing/"+video_name+".mp4")
    cmd = "ffmpeg -ss "+str(start_time)+" -i \""+url+"\" -ss 10 -t 5 C:/Users/Dell/Desktop/Presentation/Results/testing/"+video_name+".mp4"
    os.system(cmd)
    #DETACHED_PROCESS = 0x00000008
    #subprocess.call('taskkill /F /IM exename.exe', creationflags=DETACHED_PROCESS)    
    #subprocess.run(cmd)


def test_on_youtube_video(video_link):
    url,duration = get_url_to_download(video_link)
    if url == -1:
        print(" Not succesful")
    else:
        if duration > min_video_duration+10 and duration < 2400  :
            print(duration)
            
            start_times = sorted(random.sample(range(min_video_duration, duration-10), 1))
            for j in range(1):
                name =  'input'
                download_video(url,start_times[j],name)
            
    # =============================================================================
    #         name =  '{0:03d}'.format(k)
    #         start_times = random.sample(range(min_video_duration, duration-10), 1)
    #         download_video(url,start_times[0],name)
    #         k = k+1
    # =============================================================================
