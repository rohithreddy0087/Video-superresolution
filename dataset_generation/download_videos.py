# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 17:09:45 2020

@author: Dell
"""

import pandas as pd
import youtube_dl
import os,subprocess
import random
pd.options.mode.chained_assignment = None  # default='warn'
ifile = open('youtube_video_links_deniz.csv')
df = pd.read_csv(ifile,delimiter = ';')

video_links = df['Video_link']

video_height = 144
video_width = 256
video_res = '144p'
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
    cmd = "ffmpeg -ss "+str(start_time)+" -i \""+url+"\" -ss 10 -t 5 C:/Users/Dell/Desktop/BTP/RBPN/rbpn/dataset/deniz_dataset/"+video_name+".mp4"
    os.system(cmd)
    #DETACHED_PROCESS = 0x00000008
    #subprocess.call('taskkill /F /IM exename.exe', creationflags=DETACHED_PROCESS)    
    #subprocess.run(cmd)


k = 1
for i in range(len(video_links)):
    url,duration = get_url_to_download(video_links[i])
    if url == -1:
        print(str(i)+" Not succesful")
    else:
        if duration > min_video_duration+10 and duration < 2400  :
            print(duration)
            
            start_times = sorted(random.sample(range(min_video_duration, duration-10), 3))
            
            for j in range(3):
                name =  '{0:03d}'.format(k)
                download_video(url,start_times[j],name)
                k = k+1
            """
            name =  '{0:03d}'.format(k)
            start_times = random.sample(range(3, 15), 1)
            download_video(url,start_times[0],name)
            k = k+1
            """