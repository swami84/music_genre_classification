# Music Genre Classification

## Introduction

Humans can identify genre by listening to the song very shortly. While most of us can agree with each other on our genre label for a given song/music but it becomes hard to explain on how our minds processed the song and came to the decision
The even harder problem is to generalize our process and translate it to a machine-ready algorithm. Here I take an effort to classify top genres as listed on last.fm website by training 2 models - using Convolutional Neural Network (CNN) and Convolution Recurrent Neural Network(CRNN) models.

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

#### Accuracy

| Model | Accuracy |
| ----- | -------- |
| CNN   | 65.81%   |
| CRNN  | 64.14%   |

#### Confusion Matrix

![alt text](https://github.com/swami84/music_genre_classification/blob/master/data/images/model_comparison_norm_heatmap.jpg)

## References

1. Adiyansjah, Alexander A S Gunawan, Derwin Suhartono,
   Music Recommender System Based on Genre using Convolutional Recurrent Neural Networks,
   Procedia Computer Science,
   Volume 157, 2019, Pages 99-109, ISSN 1877-0509,
   https://doi.org/10.1016/j.procs.2019.08.146.
   (https://www.sciencedirect.com/science/article/pii/S1877050919310646)
2. 









