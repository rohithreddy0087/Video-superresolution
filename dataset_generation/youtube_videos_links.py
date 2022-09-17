# -*- coding: utf-8 -*-
"""
Created on Mon Aug 10 15:29:14 2020

@author: Dell
"""
import urllib.request
import json,csv

api_key = 'AIzaSyAbkvTw4livBaz6PAv9wSNWoNN9ti7Z4eY'
#channel_id = "UCP1UlYJH_QL4m5HVyikcxfQ"
channel_id = "UCN_zEeX1PVvk8kAQierYo3g"
def get_all_video_in_channel(channel_id):
    #api_key = YOUR API KEY

    base_video_url = 'https://www.youtube.com/watch?v='
    base_search_url = 'https://www.googleapis.com/youtube/v3/search?'

    first_url = base_search_url+'key={}&channelId={}&part=snippet,id&order=date&maxResults=25'.format(api_key, channel_id)

    video_links = []
    url = first_url
    while True:
        inp = urllib.request.urlopen(url)
        resp = json.load(inp)

        for i in resp['items']:
            if i['id']['kind'] == "youtube#video":
                video_links.append(base_video_url + i['id']['videoId'])

        try:
            next_page_token = resp['nextPageToken']
            url = first_url + '&pageToken={}'.format(next_page_token)
        except:
            break
    #print(resp)
    return video_links

s = get_all_video_in_channel(channel_id)

csvfile_data = open('youtube_video_links_deniz.csv', 'w')
csvwriter = csv.writer(csvfile_data, delimiter=';')
csvfile_head = ['S.no','Video_link']
csvwriter.writerow(csvfile_head)
for i in range(len(s)):
    csvfile = []
    csvfile.append(i+1)
    csvfile.append(s[i])
    csvwriter.writerow(csvfile)