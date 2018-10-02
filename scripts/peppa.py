
import pandas as pd
import numpy as np
import requests
import os


import matplotlib.pyplot as plt
import seaborn as sns
from apiclient.discovery import build
from apiclient.errors import HttpError
#from google.auth.tools import argparser
from youtube_transcript_api import YouTubeTranscriptApi

f = open('/Users/nitya/youtopian/youtube', 'r')
DEVELOPER_KEY = f.read().strip()
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"
youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION,developerKey=DEVELOPER_KEY)


def get_video_id(q, max_results,token, order="relevance",
                 location=None, location_radius=None):

    search_response = youtube.search().list(
        q=q, type="video", pageToken=token,part="id,snippet",
        maxResults=max_results, location=location,
        locationRadius=location_radius, safeSearch = 'none').execute()
    videoId = []
    title = []
    description = []
    tok = search_response['nextPageToken']

    for search_result in search_response.get("items", []):
        if search_result["id"]["kind"] == "youtube#video":
            title.append(search_result['snippet']['title'])
            videoId.append(search_result['id']['videoId'])
            response = youtube.videos().list(part='statistics, snippet',
                                             id=search_result['id']['videoId']
                                            ).execute()
            description.append(response['items'][0]['snippet']['description'])

    ydict = {'title':title,'videoId':videoId,
              'description':description}
    return ydict, tok

def keyword_search(keyword, results_enter):

    tests, token0 = get_video_id(keyword, 50,order = "relevance",
                                 token= None)
    newdf = pd.DataFrame(tests)
    #Total number of results I need..:
    results = results_enter-50

    while results>50:
        tests, token0 = get_video_id(keyword, 50,order = "relevance",
                                     token= str(token0))
        df = pd.DataFrame(tests)
        newdf = pd.concat([newdf,df]).reset_index(drop=True)
        results = results-50

    filename = keyword.replace(" ", "_") + '_' + str(results_enter)
    newdf.to_csv(filename)

    return


# -- Getting Comments -----

def get_comment_threads(video_id, results):
    results = youtube.commentThreads().list(
    part="snippet",
    videoId=video_id,
    textFormat="plainText",
    maxResults = results
    ).execute()
    for item in results["items"]:
        comment = item["snippet"]["topLevelComment"]
        author = comment["snippet"]["authorDisplayName"]
        text = comment["snippet"]["textDisplay"]
        #print "Comment by %s: %s" % (author, text)
    ydict = {'videoId':videoId, 'Comment':comment}
    return ydict


# Call the API's comments.list method to list the existing comment replies.
def get_comments(youtube, parent_id):
    results = youtube.comments().list(
    part="snippet",
    parentId=parent_id,
    textFormat="plainText",
    maxResults=100
    ).execute()
    for item in results["items"]:
        author = item["snippet"]["authorDisplayName"]
        text = item["snippet"]["textDisplay"]
        #print "Comment by %s: %s" % (author, text)

    return results["items"]

#Search for videos by channel ID.....

def channel_snatch(q, max_results, token, channelId,
                   order="relevance"):

    search_response = youtube.search().list(
        q=q, type="video", pageToken=token,part="id,snippet",
        maxResults=max_results, channelId = channelId,
        location=None, locationRadius=None,
        safeSearch = 'strict').execute()
    videoId = []
    title = []
    description = []
    tok = search_response['nextPageToken']

    for search_result in search_response.get("items", []):
        if search_result["id"]["kind"] == "youtube#video":
           title.append(search_result['snippet']['title'])
           videoId.append(search_result['id']['videoId'])
           response = youtube.videos().list(part='statistics, snippet',
                                            id=search_result['id']['videoId']
                                            ).execute()
           description.append(response['items'][0]['snippet']['description'])

    ydict = {'title':title,'videoId':videoId,
              'description':description}
    return ydict, tok


def search_by_channel(keyword, results_enter, channelId):

    tests, token0 = channel_snatch(keyword, 50,token= None,
                                   channelId = channelId)
    newdf = pd.DataFrame(tests)
    #Total number of results I need..:
    results = results_enter-50

    while results>50:
        tests, token0 = channel_snatch(keyword, 50,
                                       token= str(token0),
                                     channelId = channelId)
        df = pd.DataFrame(tests)
        newdf = pd.concat([newdf,df]).reset_index(drop=True)
        results = results-50

    filename = keyword.replace(" ", "_") + '_' + 'channel' + str(results_enter)
    newdf.to_csv(filename)
    return



# Call the API's captions.list method to list the existing caption tracks.
def list_captions(youtube, video_id):
    results = youtube.captions().list(
    part="snippet",
    videoId=video_id
    ).execute()
    for item in results["items"]:
        id = item["id"]
        name = item["snippet"]["name"]
        language = item["snippet"]["language"]
    #print "Caption track '%s(%s)' in '%s' language." % (name, id, language)

    return results["items"]

 #transcripts using requests requests..?
