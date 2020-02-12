---
jupyter:
  jupytext:
    formats: ipynb,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.2'
      jupytext_version: 1.3.3
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Week 6


## Corpus

We are going to look at a set of three playlists. They are called:

'<b>Old School Reggae Roots 70s/80s</b>' (300 tracks)<br>
'<b>Heavy roots dub reggae</b>' (317 tracks)<br>
'<b>DEEP MEDi MUSIK & Tempa Records .. deep</b>' (617 tracks)

These are all playlists of styles of music that are somhow connected. The theory is that dub (2nd playlist) evolved from reggae (1st playlist), and dupstep (3rd playlist) evolved from dub. Therefore these three playlist represent the 'evolution of reggae into dubstep'. 

Note: We are not talking about the 'brostep' genre, commonly referred to as dubstep. brostep artists include Skrillex and Datsik.

## Preliminary Hypotheses

We want to see if we can somehow find a 'shift' in these playlists. That is, the distribution of Spotify's features of these three playlists follow some path. 

We can see for example, that reggae is traditionally played on acoustic or amplified musical intruments, dub songs are often a re-mix of the individual tracks of a reggae song, using a lot of (analogue) effects, and dubstep is usually entirely electronically/digitally produced. Because of this, we expect to see some decrease in the 'acousticness' feature in these playlists.

We can also argue that the instrumentalness will increase. Reggae usually has vocals, where dubstep typically doesn't.

TODO: MORE FEATURES

<details>
<summary>See code</summary>
<p>

Imports


```python
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials, SpotifyOauthError
from spotipy_credentials import *

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import seaborn as sns

from IPython.display import Image

import pandas as pd

sns.set(rc={'figure.figsize':(11.7,8.27)})
```

</p>
</details>

We need to use the credentials of our Spotify account to connect to the API. We use a wrapper called spotipy to do this. I've stored my credentials in a python file called `spotipy_credentials.py` (see import above). Edit that file so it contains your credentials.

<details>
<summary>See code</summary>
<p>

```python
try:
    client_credentials_manager = SpotifyClientCredentials(
        client_id = SPOTIPY_CLIENT_ID,
        client_secret = SPOTIPY_CLIENT_SECRET
    )
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)
except SpotifyOauthError:
    print("Spotipy credentials not found. Add them to `spotipy_credentials.py` in this folder")
```

</p>
</details>

We are going to look at some playlists. We want to bbe able to see the features of all tracks in a playlist. To do this we create a function to collect all these features.

<details>
<summary>See code</summary>
<p>


```python
def get_playlist_features(playlist, tracks_per_iteration = 50):
    """" func: get_playlist_features
        args:
            playlist: the result of a spotify query, specifically one of the 'items' under the 
                'playlist' field
            tracks_per_iteration: amount of tracks per iteration, recommended to keep at 50
                (values greater than 50  will not work)
        returns:
            features:
                a list of dicts containing spotify's features, one per track
    """
    n = 0
    
    features = []
    while True:
        # get the next batch of tracks
        tracks = sp.playlist_tracks(playlist["id"], limit=tracks_per_iteration, offset=n)
        
        # stop if we can't find more tracks
        if not len(tracks["items"]):
            break
        
        # find the ids of these tracks
        track_ids = []
        for i in range(len(tracks["items"])):
            # check if id is valid
            if tracks["items"][i]["track"]["id"]:
                track_ids.append(tracks["items"][i]["track"]["id"])
                
        # add the features of these tracks to the features list
        features += sp.audio_features(track_ids)
        
        # prepare for next batch
        n += tracks_per_iteration
        
    while None in features:
        features.remove(None)
        
    return features

```

</p>
</details>

Next up, we want to query spotify for our playlists and save the collected features in pandas DataFrames.

<details>
<summary>See code</summary>
<p>


```python
### Getting data from API
# Playlist 1
q1 = "old school reggae roots"
playlist1 = sp.search(q1, type="playlist", limit=1)["playlists"]["items"][0]

pl1 = playlist1['name']
print(f"Playlist analysis: {pl1}, with {playlist1['tracks']['total']} tracks")

df1 = pd.DataFrame(get_playlist_features(playlist1))


# Playlist 2
q2 = "heavy dub roots reggae"
playlist2 = sp.search(q2, type="playlist", limit=1)["playlists"]["items"][0]

pl2 = playlist2['name']
print(f"Playlist analysis: {pl2}, with {playlist2['tracks']['total']} tracks")

df2 = pd.DataFrame(get_playlist_features(playlist2))


# Playlist 3
q3 = "deep medi musik"
playlist3 = sp.search(q3, type="playlist", limit=1)["playlists"]["items"][0]

pl3 = playlist3['name']
print(f"Playlist analysis: {pl3}, with {playlist3['tracks']['total']} tracks")

df3 = pd.DataFrame(get_playlist_features(playlist3))

df3.head()
```

</p>
</details>

Now that we've loaded all the features in our playlist, we want to visually analyze the results. We define a list with all the features we want to look at.

<details>
<summary>See code</summary>
<p>

```python
# list of features to look at, including the range these values will be in (used later)
interesting_features = [
    ['danceability', 0, 1],
    ['energy', 0, 1],
    ['speechiness',0, 1],
    ['acousticness', 0, 1],
    ['duration_ms', 0, 1000000],
    ['instrumentalness', 0, 1],
    ['liveness', 0, 1],
    ['valence', 0, 1],
    ['tempo', 20, 220],
    ['loudness', -20, 0]
]
```

</p>
</details>

For each feature, we make a plot showing the distribution for each playlist, also showing the mean of the data

<details>
<summary>See code</summary>
<p>


```python
### Plots
sns.set(rc={'figure.figsize':(13,30)})
fig, axs = plt.subplots(5, 2)

for i in range(5):
    for j in range(2):
        feature, minv, maxv = interesting_features[i + (5*j)]
        # define the bin width, so the bins will be the same size for the 
        # different playlists
        binwidth = abs(maxv - minv)/20
        
        sns.distplot(
            df1[feature], label=pl1, ax=axs[i, j], color="#ae1d26",
            # calculate amount of bins based on binwidth defined earlier:
            bins=int(abs(df1[feature].max() - df1[feature].min())/binwidth), 
            kde=False, norm_hist=True
        )
        sns.distplot(
            df2[feature], label=pl2, ax=axs[i, j], color="#f0c435",
            # calculate amount of bins based on binwidth defined earlier:
            bins=int(abs(df2[feature].max() - df2[feature].min())/binwidth),
            kde=False, norm_hist=True
        )
        sns.distplot(
            df3[feature], label=pl3, ax=axs[i, j], color="#275d2e",
            # calculate amount of bins based on binwidth defined earlier:
            bins=int(abs(df3[feature].max() - df3[feature].min())/binwidth),
            kde=False, norm_hist=True
        )
        
        # represent the means of the data as vertical dashed lines
        l1 = axs[i, j].axvline(df1[feature].mean(), ls='--', color="#ae1d26")
        l2 = axs[i, j].axvline(df2[feature].mean(), ls='--', color="#f0c435")
        l3 = axs[i, j].axvline(df3[feature].mean(), ls='--', color="#275d2e")
        axs[i, j].legend(
            (l1, l2, l3), (
                f"mean={df1[feature].mean():.2f}, std={df1[feature].std():.2f}",
                f"mean={df2[feature].mean():.2f}, std={df2[feature].std():.2f}",
                f"mean={df3[feature].mean():.2f}, std={df3[feature].std():.2f}"
            )
        )

handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')
fig.tight_layout()
plt.subplots_adjust(hspace = 0.3, top=0.95)
```

</p>
</details>

From visual inspection, we can already draw some preliminary conclusions. We can see that our hypotheses had some meaning. 

The acousticness does seem to decrease, with the mean going from 0.24 $\rightarrow$ 0.13 $\rightarrow$ 0.06. 

The intrumentalness also has a remarkable path. Reggae has a very low score (mean of 0.09), meaning it has a lot of vocals. Dub is significantly more instrumental with a mean score of 0.46, Dubstep is even more intrumental with a mean score of 0.65.

TODO: MORE FEATURES


### Statistical Significance

To verify our conclusions we have to look at some degree of statistical significance. For now we will look at how 'far', that is, how many standard deviations the mean of the scores for one playlist lies from another. We check all of these at once and store them in a table.

Here, negative values mean 'less' and positive values mean 'more' (negative value can mean: reggae is less danceable than dub).

<details>
<summary>See code</summary>
<p>


```python
### Significance

# take the means and standard deviations, and make those values the columns so we can 
# perform vector operations on them
agg1 = df1.agg(["mean", "std"]).T
agg2 = df2.agg(["mean", "std"]).T
agg3 = df3.agg(["mean", "std"]).T

# check how many standard deviations two genres lie from eachother
significance = pd.DataFrame({
    "reggae - dub":     (agg1["mean"] - agg2["mean"]) / agg1["std"],
    "reggae - dubstep": (agg1["mean"] - agg3["mean"]) / agg1["std"],
    "dub - reggae":     (agg2["mean"] - agg1["mean"]) / agg2["std"],
    "dub - dubstep":    (agg2["mean"] - agg3["mean"]) / agg2["std"],
    "dubstep - reggae": (agg3["mean"] - agg1["mean"]) / agg3["std"],
    "dubstep - dub":    (agg3["mean"] - agg2["mean"]) / agg3["std"]
})

significance
```

</p>
</details>

Here we can see our preliminary conclusions were infact somewhat meaningful.

Dub is somewhat less acoustic than reggae (value of -0.61, meaning 0.61 standard deviations below the mean), and dubstep is somewhat less acoustic than dub (value of -0.56).

Reggae is a lot less instrumental than dub (value of -1.70), and dub is a somewhat less intrumental than dubstep (value of -0.50).

TODO: MORE FEATURES


### Iconic tracks

Just for fun, we are also going to look at what the most typical/iconic tracks are according to each playlist. We'll do this by calculating some distance measure for each track to the 'golden standard' for that playlist, defined by the feature vector consisting of the means.

We will use euclidean distance

<details>
<summary>See code</summary>
<p>


```python
distance_features = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence"
]

def most_iconic_track(df):
    golden_standard = df[distance_features].mean()

    # euclidean distance:
    df["distance"] = (df[distance_features] - golden_standard).pow(2).sum(axis=1).pow(0.5)
    closest_song_id = df[df['distance'] == df['distance'].min()]["id"].values[0]

    closest_track = sp.track(closest_song_id)
    return closest_track['name'], closest_track['artists'][0]['name']

def least_iconic_track(df):
    golden_standard = df[distance_features].mean()

    # euclidean distance:
    df["distance"] = (df[distance_features] - golden_standard).pow(2).sum(axis=1).pow(0.5)
    furthest_song_id = df[df['distance'] == df['distance'].max()]["id"].values[0]

    furthest_track = sp.track(furthest_song_id)
    return furthest_track['name'], furthest_track['artists'][0]['name']


track, artist = most_iconic_track(df1)
print(f"The most iconic reggae track is: '{track}' by '{artist}'")

track, artist = most_iconic_track(df2)
print(f"The most iconic dub track is: '{track}' by '{artist}'")

track, artist = most_iconic_track(df3)
print(f"The most iconic dubstep track is: '{track}' by '{artist}'")

print('\n')

track, artist = least_iconic_track(df1)
print(f"The least iconic reggae track is: '{track}' by '{artist}'")

track, artist = least_iconic_track(df2)
print(f"The least iconic dub track is: '{track}' by '{artist}'")

track, artist = least_iconic_track(df3)
print(f"The least iconic dubstep track is: '{track}' by '{artist}'")

```

</p>
</details>

<details>
<summary>See code</summary>
<p>


```python
def show_most_iconic(query):
    playlists = sp.search(query, type="playlist", limit=20)["playlists"]["items"]
    
    playlist = None
    maxtracks = 0
    for i in range(len(playlists)):
        if playlists[i]['tracks']['total'] > maxtracks:
            maxtracks = playlists[i]['tracks']['total']
            playlist = playlists[i]

    print(f"Playlist analysis: {playlist['name']}, with {playlist['tracks']['total']} tracks")
    df = pd.DataFrame(get_playlist_features(playlist))

    track, artist = most_iconic_track(df)
    print(f"The most iconic track is: '{track}' by '{artist}'")
    
    fig, axs = plt.subplots(5, 2)
    for i in range(5):
        for j in range(2):
            feature, _, _ = interesting_features[i + (5*j)]
            sns.distplot(df[feature], ax=axs[i, j], bins=10, kde=False, norm_hist=True)

show_most_iconic("pop")
```

</p>
</details>

```python

```

```python

```
