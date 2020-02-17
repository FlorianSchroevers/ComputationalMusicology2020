
# Evolution of Reggae into Dub into Dubstep

 - Florian Schroevers
 - 11334266
 - Feb 2020

Note: The code doesn't preview properly in this readme. See the main notebook in the repository or visit florianschroevers.github.io/ComputationalMusicology2020

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
First we need get our corpus from the Spotify API. In a file called `helpers.py`, we've defined some helper functions to load all this data in batches, to keep the amount of code in this notebook to a minimum. 

First we need to do some imports, then we will load the track features of the  playlists and save them in pandas dataframes.

<details>
<summary>See code</summary>
<p>

{% highlight python %}
### imports
import matplotlib.pyplot as plt
import seaborn as sns

import helpers
{% endhighlight %}


{% highlight python %}
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

{% endhighlight %}

</p>
</details>

    Playlist analysis: Old School Reggae Roots 70s/80s, with 299 tracks
    Playlist analysis: Heavy Dub Roots Reggae, with 835 tracks
    Playlist analysis: DEEP MEDi MUSIK & Tempa Records .. deep, with 617 tracks

## Visualization

We want to have a good idea of what our corpus looks like. We will generate some histograms of the three playlists for each feature. We will also plot the means for each playlist. This should give us not only an idea of how the features playlists are distributed, but also how the different playlists relate.

<details>
<summary>See code</summary>
<p>

{% highlight python %}
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
{% endhighlight %}
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

{% highlight python %}
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
{% endhighlight %}

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

### Clustering

We want some way to actually see how these playlists are represented by the track features. If we see the feature vectors as points in 7-dimensional space, we can create a scatterplot showing all tracks. Of course we can't see 7 dimensions, so we need to perform some dimensionality redution. There is an array of methods to use. Based on some trial and error, the t-SNE method was chosen. (See clustering_viz.ipynb for an example of all the methods).

<details>
<summary>See code</summary>
<p>

{% highlight python %}

from sklearn.manifold import *

features = [
    "danceability",
    "energy",
    "speechiness",
    "acousticness",
    "instrumentalness",
    "liveness",
    "valence"
]

df1["playlist"] = "reggae"
df2["playlist"] = "dub"
df3["playlist"] = "dubstep"
complete_df = pd.concat([df1, df2, df3])

tsne = TSNE(n_components=2, perplexity=80)
principal_components = tsne.fit_transform(complete_df[features].values)
principal_df = pd.DataFrame(data=principal_components, columns = ['pc1', 'pc2'])

complete_df["pc1"] = principal_components[:, 0]
complete_df["pc2"] = principal_components[:, 1]

sns.set_style("white")
sns.set(rc={
    'figure.figsize': (13,13),
    'axes.facecolor': 'darkgrey',
    'figure.facecolor': 'white',
    'axes.grid': False
})
sns.scatterplot(data=complete_df, x="pc1", y="pc2", hue="playlist", 
                palette = ["#00992f", "#f7ee00", "#eb0000"])

{% endhighlight %}

</p>
</details>

![png](figs/figures.png)


### Iconic tracks

Just for fun, we are also going to look at what the most typical/iconic tracks are according to each playlist. We'll do this by calculating some distance measure for each track to the 'golden standard' for that playlist, defined by the feature vector consisting of the means.

We will use euclidean distance

<details>
<summary>See code</summary>
<p>

{% highlight python %}

def most_iconic_track(df):
    golden_standard = df[features].mean()

    # euclidean distance:
    df["distance"] = (df[features] - golden_standard).pow(2).sum(axis=1).pow(0.5)
    closest_song_id = df[df['distance'] == df['distance'].min()]["id"].values[0]

    closest_track = sp.track(closest_song_id)
    return closest_track['name'], closest_track['artists'][0]['name']

def least_iconic_track(df):
    golden_standard = df[features].mean()

    # euclidean distance:
    df["distance"] = (df[features] - golden_standard).pow(2).sum(axis=1).pow(0.5)
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

{% endhighlight %}

</p>
</details>


    The most iconic reggae track is: 'Long Shot Bus Me Bet' by 'The Pioneers'
    The most iconic dub track is: 'Taxi to Baltimore Dub' by 'Scientist'
    The most iconic dubstep track is: 'Wobble That Gut' by 'Skream'
    
    
    The least iconic reggae track is: 'Satta Dub' by 'King Tubby'
    The least iconic dub track is: 'Conquering Lion - Dub Plate Mix' by 'Yabby You'
    The least iconic dubstep track is: 'A Song For Lenny' by 'Skream'

