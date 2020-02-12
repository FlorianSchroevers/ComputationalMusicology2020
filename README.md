
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

## Setting up
First we need...

<details>
<summary>See code</summary>
<p>

```python
### imports
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import seaborn as sns

from helpers import *
```


```python
# Playlist 1
p1_name, p1_tracks = collect_tracks_query("old school reggae roots", "playlist")
df1 = get_tracklist_features(p1_tracks)
print(f"Playlist analysis: {p1_name}, with {len(p1_tracks)} tracks")

# Playlist 1
p2_name, p2_tracks = collect_tracks_query("heavy dub roots reggae", "playlist")
df2 = get_tracklist_features(p2_tracks)
print(f"Playlist analysis: {p2_name}, with {len(p2_tracks)} tracks")

# Playlist 1
p3_name, p3_tracks = collect_tracks_query("deep medi musik", "playlist")
df3 = get_tracklist_features(p3_tracks)
print(f"Playlist analysis: {p3_name}, with {len(p3_tracks)} tracks")

df3.head()
```

</p>
</details>

    Playlist analysis: Old School Reggae Roots 70s/80s, with 299 tracks
    Playlist analysis: Heavy Dub Roots Reggae, with 835 tracks
    Playlist analysis: DEEP MEDi MUSIK & Tempa Records .. deep, with 617 tracks





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>danceability</th>
      <th>energy</th>
      <th>key</th>
      <th>loudness</th>
      <th>mode</th>
      <th>speechiness</th>
      <th>acousticness</th>
      <th>instrumentalness</th>
      <th>liveness</th>
      <th>valence</th>
      <th>tempo</th>
      <th>type</th>
      <th>id</th>
      <th>uri</th>
      <th>track_href</th>
      <th>analysis_url</th>
      <th>duration_ms</th>
      <th>time_signature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>0.748</td>
      <td>0.721</td>
      <td>0</td>
      <td>-8.717</td>
      <td>0</td>
      <td>0.0538</td>
      <td>0.002190</td>
      <td>0.606</td>
      <td>0.181</td>
      <td>0.504</td>
      <td>139.992</td>
      <td>audio_features</td>
      <td>26jDjZ9LXtiWlbDwRHwJD8</td>
      <td>spotify:track:26jDjZ9LXtiWlbDwRHwJD8</td>
      <td>https://api.spotify.com/v1/tracks/26jDjZ9LXtiW...</td>
      <td>https://api.spotify.com/v1/audio-analysis/26jD...</td>
      <td>294853</td>
      <td>4</td>
    </tr>
    <tr>
      <td>1</td>
      <td>0.759</td>
      <td>0.472</td>
      <td>11</td>
      <td>-17.413</td>
      <td>0</td>
      <td>0.0660</td>
      <td>0.091400</td>
      <td>0.865</td>
      <td>0.253</td>
      <td>0.917</td>
      <td>142.970</td>
      <td>audio_features</td>
      <td>1RtsJXVXDVvISNzrc0WohD</td>
      <td>spotify:track:1RtsJXVXDVvISNzrc0WohD</td>
      <td>https://api.spotify.com/v1/tracks/1RtsJXVXDVvI...</td>
      <td>https://api.spotify.com/v1/audio-analysis/1Rts...</td>
      <td>284453</td>
      <td>4</td>
    </tr>
    <tr>
      <td>2</td>
      <td>0.927</td>
      <td>0.289</td>
      <td>0</td>
      <td>-12.240</td>
      <td>1</td>
      <td>0.5530</td>
      <td>0.366000</td>
      <td>0.888</td>
      <td>0.330</td>
      <td>0.377</td>
      <td>139.971</td>
      <td>audio_features</td>
      <td>4icnCijYoAPl1DevVpatAz</td>
      <td>spotify:track:4icnCijYoAPl1DevVpatAz</td>
      <td>https://api.spotify.com/v1/tracks/4icnCijYoAPl...</td>
      <td>https://api.spotify.com/v1/audio-analysis/4icn...</td>
      <td>249467</td>
      <td>4</td>
    </tr>
    <tr>
      <td>3</td>
      <td>0.869</td>
      <td>0.576</td>
      <td>5</td>
      <td>-10.527</td>
      <td>0</td>
      <td>0.1470</td>
      <td>0.000836</td>
      <td>0.685</td>
      <td>0.411</td>
      <td>0.257</td>
      <td>147.014</td>
      <td>audio_features</td>
      <td>41GZiQAvgzSCXLhIWVJfMB</td>
      <td>spotify:track:41GZiQAvgzSCXLhIWVJfMB</td>
      <td>https://api.spotify.com/v1/tracks/41GZiQAvgzSC...</td>
      <td>https://api.spotify.com/v1/audio-analysis/41GZ...</td>
      <td>318587</td>
      <td>4</td>
    </tr>
    <tr>
      <td>4</td>
      <td>0.696</td>
      <td>0.734</td>
      <td>7</td>
      <td>-9.310</td>
      <td>1</td>
      <td>0.0533</td>
      <td>0.032100</td>
      <td>0.908</td>
      <td>0.464</td>
      <td>0.595</td>
      <td>139.953</td>
      <td>audio_features</td>
      <td>5f7ZWdkpQ05topF1yGPBwB</td>
      <td>spotify:track:5f7ZWdkpQ05topF1yGPBwB</td>
      <td>https://api.spotify.com/v1/tracks/5f7ZWdkpQ05t...</td>
      <td>https://api.spotify.com/v1/audio-analysis/5f7Z...</td>
      <td>374880</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



## Visualization

We want to have a good idea of what our corpus looks like. We will generate some histograms of the three playlists for each feature. We will also plot the means for each playlist. This should give us not only an idea of how the features playlists are distributed, but also how the different playlists relate.

<details>
<summary>See code</summary>
<p>

```python
### Vizualization
# Now that we've loaded all the features in our playlist, we want to visually analyze the 
# results. We define a list with all the features we want to look at, including the ranges
# the values will be in.
###

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

### Plots
# we will generate 10 histograms, one for each interesting feature, each histogram holds
# the distributions of that feature of each playlist, seperated by different colors.
sns.set_style("white")
sns.set(rc={
    'figure.figsize':(13,30),
    'axes.facecolor':'darkgrey',
    'figure.facecolor':'white',
    'axes.grid' : False
})
fig, axs = plt.subplots(5, 2)

for i in range(5):
    for j in range(2):
        ft, minv, maxv = interesting_features[i + (5*j)]
        # define the bin width, so the bins will be the same size for the 
        # different playlists
        binwidth = abs(maxv - minv)/25
        
        # calculate amount of bins based on binwidth defined earlier
        # the colors of eacht plot are based on the flag of ethiopia (rastafari flag)
        sns.distplot(
            df1[ft], label=p1_name, ax=axs[i, j], color="#00992f", kde=False, norm_hist=True, 
            bins=int(abs(df1[ft].max() - df1[ft].min())/binwidth)
            
        )
        sns.distplot(
            df2[ft], label=p2_name, ax=axs[i, j], color="#f7ee00", kde=False, norm_hist=True, 
            bins=int(abs(df2[ft].max() - df2[ft].min())/binwidth)
        )
        sns.distplot(
            df3[ft], label=p3_name, ax=axs[i, j], color="#eb0000", kde=False, norm_hist=True, 
            bins=int(abs(df3[ft].max() - df3[ft].min())/binwidth)
        )
        
        # represent the means of the data as vertical dashed lines
        l1 = axs[i, j].axvline(df1[ft].mean(), ls='--', color="#00992f")
        l2 = axs[i, j].axvline(df2[ft].mean(), ls='--', color="#f7ee00")
        l3 = axs[i, j].axvline(df3[ft].mean(), ls='--', color="#eb0000")
        
        # legend for each subplot displaying the means and stds
        axs[i, j].legend((l1, l2, l3), (
            f"mean={df1[ft].mean():.2f}, std={df1[ft].std():.2f}",
            f"mean={df2[ft].mean():.2f}, std={df2[ft].std():.2f}",
            f"mean={df3[ft].mean():.2f}, std={df3[ft].std():.2f}"
        ))

handles, labels = axs[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center')
fig.tight_layout()
plt.subplots_adjust(hspace = 0.3, top=0.97)
```
</p>
</details>

![png](figures/histograms.png)


From visual inspection, we can already draw some preliminary conclusions. We can see that our hypotheses had some meaning. 

The acousticness does seem to decrease, with the mean going from 0.25 $\rightarrow$ 0.13 $\rightarrow$ 0.06. 

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


<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>reggae - dub</th>
      <th>reggae - dubstep</th>
      <th>dub - reggae</th>
      <th>dub - dubstep</th>
      <th>dubstep - reggae</th>
      <th>dubstep - dub</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>danceability</td>
      <td>-0.100246</td>
      <td>0.909808</td>
      <td>0.112375</td>
      <td>1.132267</td>
      <td>-0.657607</td>
      <td>-0.730064</td>
    </tr>
    <tr>
      <td>energy</td>
      <td>-0.057328</td>
      <td>-1.097308</td>
      <td>0.045761</td>
      <td>-0.830140</td>
      <td>0.736385</td>
      <td>0.697913</td>
    </tr>
    <tr>
      <td>key</td>
      <td>-0.095321</td>
      <td>-0.050129</td>
      <td>0.097075</td>
      <td>0.046024</td>
      <td>0.048993</td>
      <td>-0.044169</td>
    </tr>
    <tr>
      <td>loudness</td>
      <td>0.504107</td>
      <td>0.040016</td>
      <td>-0.387250</td>
      <td>-0.356511</td>
      <td>-0.041626</td>
      <td>0.482762</td>
    </tr>
    <tr>
      <td>mode</td>
      <td>0.452920</td>
      <td>0.414090</td>
      <td>-0.391066</td>
      <td>-0.033527</td>
      <td>-0.359258</td>
      <td>0.033688</td>
    </tr>
    <tr>
      <td>speechiness</td>
      <td>0.118601</td>
      <td>0.286484</td>
      <td>-0.119271</td>
      <td>0.168830</td>
      <td>-0.264511</td>
      <td>-0.155006</td>
    </tr>
    <tr>
      <td>acousticness</td>
      <td>0.506433</td>
      <td>0.819964</td>
      <td>-0.612404</td>
      <td>0.379138</td>
      <td>-1.456482</td>
      <td>-0.556918</td>
    </tr>
    <tr>
      <td>instrumentalness</td>
      <td>-1.692652</td>
      <td>-2.546766</td>
      <td>0.989934</td>
      <td>-0.499522</td>
      <td>1.857210</td>
      <td>0.622856</td>
    </tr>
    <tr>
      <td>liveness</td>
      <td>-0.002463</td>
      <td>-0.772459</td>
      <td>0.002329</td>
      <td>-0.728227</td>
      <td>0.436071</td>
      <td>0.434680</td>
    </tr>
    <tr>
      <td>valence</td>
      <td>0.426230</td>
      <td>2.763428</td>
      <td>-0.373278</td>
      <td>2.046837</td>
      <td>-1.864151</td>
      <td>-1.576625</td>
    </tr>
    <tr>
      <td>tempo</td>
      <td>-0.193445</td>
      <td>-0.705584</td>
      <td>0.234099</td>
      <td>-0.619767</td>
      <td>1.256071</td>
      <td>0.911702</td>
    </tr>
    <tr>
      <td>duration_ms</td>
      <td>-0.134731</td>
      <td>-1.689771</td>
      <td>0.135171</td>
      <td>-1.560118</td>
      <td>1.718617</td>
      <td>1.581586</td>
    </tr>
    <tr>
      <td>time_signature</td>
      <td>-0.057439</td>
      <td>0.435356</td>
      <td>0.058581</td>
      <td>0.502593</td>
      <td>-0.122973</td>
      <td>-0.139198</td>
    </tr>
  </tbody>
</table>
</div>



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


    The most iconic reggae track is: 'Long Shot Bus Me Bet' by 'The Pioneers'
    The most iconic dub track is: 'Taxi to Baltimore Dub' by 'Scientist'
    The most iconic dubstep track is: 'Wobble That Gut' by 'Skream'
    
    
    The least iconic reggae track is: 'Satta Dub' by 'King Tubby'
    The least iconic dub track is: 'Conquering Lion - Dub Plate Mix' by 'Yabby You'
    The least iconic dubstep track is: 'A Song For Lenny' by 'Skream'

