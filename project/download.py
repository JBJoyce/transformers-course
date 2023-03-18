import os

from pytube import YouTube, Playlist

p = Playlist("https://youtube.com/playlist?list=PLfYUBJiXbdtSvpQjSnJJ_PmDQB_VyT5iU")

print(f'Downloading: {p.title} into {os.getcwd()}')
for video in p.videos:
    video.streams.filter(only_audio=True).first().download()
