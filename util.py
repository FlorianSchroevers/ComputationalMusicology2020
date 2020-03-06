import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOauthError
import chart_studio
from credentials import *

import pandas as pd
import numpy as np

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
    piano chords
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


# +
def get_segment_interval_feature_matrix(segments, start_time, stop_time, 
                                        feature="pitches", scale_width=False):
    features_array = np.zeros(shape=(1, 12))
    
    width = 1
    n = 0
    s = segments[0]
    while s["start"] < stop_time:
        if s["start"] > start_time:
            if scale_width:
                width = int(10/s["duration"])
            new_features = np.array([s[feature]] * width)
            features_array = np.concatenate([features_array, new_features])

        n += 1
        s = segments[n]
    
    return features_array[1:].T


def get_feature_matrix(track_id, feature="pitches", start_bar=0, print_time=False,
                       n_bars=None, scale_width=False, resolution="segments"):
    analysis = sp.audio_analysis(track_id)
    segments = analysis["segments"]
    bars = analysis["bars"]

    start_time = bars[start_bar]["start"]
    
    if n_bars == None:
        n_bars = len(bars) - start_bar - 1
    stop_time = bars[start_bar + n_bars]["start"]
    
    if print_time:
        print(f"{start_time:.1f} - {stop_time:.1f}")
    
    if resolution == "segments":
        features_array = get_segment_interval_feature_matrix(
            segments, 
            start_time, 
            stop_time,
            feature=feature,
            scale_width=scale_width
        )
    elif resolution == "beats":
        beats = analysis["beats"]
        features_array = np.zeros(shape=(1, 12))
        
        n = 0
        b = beats[0]
        while b["start"] < stop_time:
            if b["start"] >= start_time:
                beat_mean_feature = np.array([get_segment_interval_feature_matrix(
                    segments,
                    b["start"],
                    b["start"] + b["duration"],
                    feature=feature,
                    scale_width=True
                ).mean(axis=1)])
                features_array = np.concatenate([features_array, beat_mean_feature])
            n += 1
            b = beats[n]
            
        features_array = features_array[1:].T
    
    return features_array
# -


