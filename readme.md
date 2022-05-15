# Music Genre Classification

## Introduction

Humans can identify genre by listening to the song very shortly. While most of us can agree with each other on our genre label for a given song/music but it becomes hard to explain on how our minds processed the song and came to the decision
The even harder problem is to generalize our process and translate it to a machine-ready algorithm. Here I take an effort to classify top genres as listed on last.fm website by training 2 models - using Convolutional Neural Network (CNN) and Convolution Recurrent Neural Network(CRNN) models.

## Data

Top 100 songs for each genre (total 17) were scraped from [last.fm](https://www.last.fm/music)

![alt text](https://github.com/swami84/music_genre_classification/blob/master/data/images/last_fm_homepage.jpg)

### Data Wrangling

- Indie, Blues, Alternative, 80s, British genres were removed
- hardcore and metal genres were combined together as metal
- hip-hop and rap genres were combined together as hip-hop
- rock and punk genres were combined together as rock

### Feature Engineering

Music genres have distinct signatures that can be often seen in their frequency response spectra. This frequency response is then converted into a non-linear scale  which is fed as the input to the deep learning models used in this work. Below we can see how the Mel spectrogram is different for songs in different genres . 

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

CNN Model

![alt text](https://github.com/swami84/music_genre_classification/blob/master/data/images/cnn_model.jpg)

CRNN Model

![alt text](https://github.com/swami84/music_genre_classification/blob/master/data/images/crnn_model.jpg)



## Results

#### Accuracy

| Model | Accuracy |
| ----- | -------- |
| CNN   | 68.49%   |
| CRNN  | 63.37%   |

#### Confusion Matrix

![alt text](https://github.com/swami84/music_genre_classification/blob/master/data/images/model_comparison_norm_heatmap.jpg)

- CNN model out performs CNN model when comparing accuracy of predictions on test dataset
- Both models are able to classify classical songs very accurately
- CNN model is able to classify reggae genre with higher accuracy compared to CRNN model
- Both models perform poorly (<50% accuracy) on predicting genre for songs belonging to Electronic and dance
- CNN model also performs poorly on predicting genre for country songs
- Both models show inter-class misclassification between Electronic - Dance and Rock - Metal genres
  - Due to similar nature of songs/style between the genres

### App

To try out the app first install the packages as

```bash
$ pip install -r requirements.txt
```

and then run 

```bash
$ streamlit run app.py
```

The app will open in the browser as shown below

![alt text](https://github.com/swami84/music_genre_classification/blob/master/data/images/Streanlit_SS_Input.jpg)

Input the youtube link for the song of interest. The app will take a few mins to download the song and process it. After that it should display a genre as shown below

![alt text](https://github.com/swami84/music_genre_classification/blob/master/data/images/Streanlit_SS_Output.jpg)

## References

1. Adiyansjah, Alexander A S Gunawan, Derwin Suhartono,
   Music Recommender System Based on Genre using Convolutional Recurrent Neural Networks,
   Procedia Computer Science,
   Volume 157, 2019, Pages 99-109, ISSN 1877-0509,
   https://doi.org/10.1016/j.procs.2019.08.146.
   (https://www.sciencedirect.com/science/article/pii/S1877050919310646)
2. @Music Genre Classifier{github,
    author={https://github.com/ericzacharia},
    title={[MusicGenreClassifier](https://github.com/ericzacharia/MusicGenreClassifier)},
    year={2022},
    url={https://github.com/ericzacharia/MusicGenreClassifier},
   }









