# Music Genre Classification

## Introduction

Human can identify genre by listening to the song very shortly. While most of us can converge on our genre identification for a given song/music but it becomes hard to explain on how we came to the conclusion. 
The even harder problem is to generalize and translate it to a machine-ready algorithm. Here I take an effort to classify top genres as listed on last.fm website by training 2 models - using Convolutional Neural Network (CNN) and Convolution Recurrent Neural Network(CRNN) models.

## Data

Top 100 songs for each genre (total 17) were scraped from [last.fm](https://www.last.fm/music)

![alt text](https://github.com/swami84/music_genre_classification/blob/master/data/images/last_fm_homepage.jpg)



### Data Wrangling



### Feature Engineering

Mel Spectrogram

![alt text](https://github.com/swami84/music_genre_classification/blob/master/data/images/genre_spectrograms.jpg)

## Model

### Zero padding

```python
if song.shape[1] >2881:
        song = song[:,:2881]
elif song.shape[1] <2881:
        song = np.pad(song, ((0,0),(0,2881-song.shape[1])), 'constant')
```

### Scaling

```python
scaler = MinMaxScaler(feature_range=(0, 1))
melspectrogram=X.reshape(X.shape[0],-1)
scaler.fit(melspectrogram)
normalized_melspectrogram = scaler.transform(melspectrogram)

features_convolution = np.reshape(normalized_melspectrogram,(X.shape[0],128, -1,1))
```

### Model Structures


## Results



![alt text](https://github.com/swami84/music_genre_classification/blob/master/data/images/model_comparison_norm_heatmap.jpg)
