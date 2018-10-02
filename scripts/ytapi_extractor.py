
import pandas as pd
import numpy as np
import requests
import os
from apiclient.discovery import build
from apiclient.errors import HttpError

f = open('/Users/nitya/youtopian/youtube', 'r')
DEVELOPER_KEY = f.read().strip()

YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,
                developerKey=DEVELOPER_KEY)


def video_query(earl):
    video_id = earl.split('=')[-1]
    return video_id

def videoId_data(video_id):
    '''
    Use the API to get data
    for a single video
    by videoId
    '''
    video_response = youtube.videos().list(id=video_id,
                                           part='snippet').execute()
    title = video_response.get('items')[0]['snippet']['title']
    description = video_response.get('items')[0]['snippet']['description']
    return title, description
