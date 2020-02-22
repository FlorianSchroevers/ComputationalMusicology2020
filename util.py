import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOauthError
import chart_studio
from credentials import *

import pandas as pd

from pprint import pprint

# ## connecting to spotify API
# We need to use the credentials of our Spotify account to connect to the API. We use a wrapper
# called spotipy to do this. I've stored my credentials in a python file called 
# `credentials.py` (see import above). Edit that file so it contains your credentials.
try:
    chart_studio.tools.set_credentials_file(username=CHART_STUDIO_USERNAME, api_key=CHART_STUDIO_API_KEY)
except:
    print("Chart studio credentials not found. Add them to `credentials.py` in this folder")

try:
    client_credentials_manager = SpotifyClientCredentials(
        client_id = SPOTIPY_CLIENT_ID,
        client_secret = SPOTIPY_CLIENT_SECRET
    )
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
except SpotifyOauthError:
    print("Spotipy credentials not found. Add them to `credentials.py` in this folder")


def get_album_tracks(album):
    tracks = []
    batch_size = 50
    
    offset = 0
    while True:
        new_tracks = sp.album_tracks(album["id"], limit=batch_size, offset=offset)["items"]
        if not len(new_tracks):
            break
        tracks += new_tracks
        offset += batch_size
    return tracks

def get_playlist_tracks(playlist):
    tracks = []
    batch_size = 100
    
    offset = 0
    while True:
        new_tracks = sp.playlist_tracks(playlist["id"], limit=batch_size, offset=offset)["items"]
        if not len(new_tracks):
            break
        tracks += [t["track"] for t in new_tracks]
        offset += batch_size
    return tracks

def collect_tracks_query(query, t):
    search_result = sp.search(query, 1, 0, t)
    tracks = []

    if "albums" in search_result:
        name = search_result["albums"]["items"][0]["name"]
        link = search_result["albums"]["items"][0]["external_urls"]["spotify"]
        tracks += get_album_tracks(search_result["albums"]["items"][0])
    elif "artists" in search_result:
        name = search_result["artists"]["items"][0]["name"]
        link = search_result["artists"]["items"][0]["external_urls"]["spotify"]
        albums = sp.artist_albums(search_result["artists"]["items"][0]["id"])
        for album in albums["items"]:
            tracks += get_album_tracks(album)
    elif "playlists" in search_result:
        name = search_result["playlists"]["items"][0]["name"]
        link = search_result["playlists"]["items"][0]["external_urls"]["spotify"]
        tracks += get_playlist_tracks(search_result["playlists"]["items"][0])
    
    return name, tracks, link

def collect_tracks_id(item_id, t):
    tracks = []
    
    if t == "album":
        album = sp.album(item_id)
        name = album["name"]
        tracks += get_album_tracks(album)
    elif t == "artist":
        name = sp.artist(item_id)["name"]
        albums = sp.artist_albums(item_id)
        for album in albums["items"]:
            tracks += get_album_tracks(album)
    elif "playlists" in search_result:
        playlist = sp.playlist(item_id)
        name = playlist["name"]
        tracks += get_playlist_tracks(playlist)

    return name, tracks

def get_tracklist_features(tracks, source=""):
    track_ids = []
    track_names = []
    for t in tracks:
        tid = t["id"]
        if tid:
            track_ids.append(t["id"])
            track_names.append(f"{t['artists'][0]['name']} - {t['name']}")
    
    batch_size = 50
    offset = 0
    
    features = []
    while offset + batch_size <= len(track_ids):
        nf = sp.audio_features(track_ids[offset:offset+batch_size])
        for i, f in enumerate(nf):
            f["name"] = track_names[offset+i]
        features += nf
        offset += batch_size
    
    features += sp.audio_features(track_ids[offset:])
    return pd.DataFrame(features)

def wrap_spotify_link_track(track):
    tname = track['name']
    artist = track['artists'][0]['name']
    link = track["external_urls"]["spotify"]
    return f'<a href="{link}">{tname} by {artist}</a>'
                               
def wrap_spotify_link(name, link):
    return f'<a href="{link}">{name}</a>'
