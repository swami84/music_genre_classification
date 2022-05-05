import streamlit as st
from tempfile import mktemp
from pydub import AudioSegment
from youtube_dl import YoutubeDL
from scipy.io import wavfile
import librosa as librosa
import librosa.display
import os
import joblib
import numpy as np
from tensorflow import keras
from keras.layers import Input,Conv2D, BatchNormalization, Dense, LSTM,MaxPooling2D
from keras.layers import Reshape, Bidirectional, LSTM,Flatten, Dropout, Activation
import time

st.set_page_config(
    page_title="Music Genre Classification", layout="centered"
)

def create_model(input_shape=(128, 2881, 1), num_classes=11, model_type='CNN', compile_model=False):
    model = keras.Sequential()

    def step(i, dim, pad, model):
        if i == 0:
            model.add(Conv2D(dim, kernel_size=(3, 3), input_shape=input_shape, name='First_Convolution'))
        else:
            model.add(Conv2D(dim, kernel_size=(3, 3)))

        model.add(BatchNormalization(axis=3))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pad, padding='same'))
        model.add(Dropout(0.1))

        return model

    layer_dims = [64, 128, 128, 128]
    pads = [(2, 2), (3, 3), (4, 4), (4, 4)]
    for (i, dim), pad in zip(enumerate(layer_dims), pads):
        model = step(i, dim, pad, model)
    if model_type == 'CNN':
        model.add(Flatten())
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
    if model_type == 'CRNN':
        fin_layer_shape = model.layers[-1].output_shape
        model.add(Reshape((fin_layer_shape[3], fin_layer_shape[2])))
        model.add(Bidirectional(LSTM(128, input_shape=(1, 128, 30), return_sequences=True, )))
        model.add(Bidirectional(LSTM(64, input_shape=(1, 128, 30), return_sequences=False)))
        model.add(Dense(num_classes))
        model.add(Activation('softmax'))
    if compile_model:
        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])
    return model
def download_clip(url, fname):
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': fname,
        'noplaylist': True,
        'continue_dl': True,
        'quiet': True,
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '320',
        }]
    }
    try:
        with YoutubeDL(ydl_opts) as ydl:
            ydl.cache.remove()
            info_dict = ydl.extract_info(url, download=False)
            ydl.prepare_filename(info_dict)
            ydl.download([url])
            return fname
    except Exception:
        return 'Error'



def generate_mels(fname):
    sound = AudioSegment.from_file(fname)
    wname = mktemp('.wav')
    sound.export(wname, format="wav")
    FS, data = wavfile.read(wname)
    corr = (48000/FS)
    n_param = int(1000/corr)
    single_chan_data = np.array(data, dtype=np.float32)

    if len(data.shape) == 2:
        single_chan_data = np.array(data[:, 0], dtype=np.float32)

    if (len(single_chan_data)/FS) >60:
        song_part_data = single_chan_data[0: 60 * FS]
    else:
        song_part_data =single_chan_data
    S = librosa.feature.melspectrogram(y=song_part_data, sr= FS, n_fft=n_param,n_mels=128,
                                       win_length=n_param,hop_length=n_param,fmax = 20000)
    S_dB = librosa.power_to_db(S, ref=np.max)
    if S_dB.shape[1] > 2881:
        S_dB = S_dB[:, :2881]
    elif S_dB.shape[1] < 2881:
        S_dB = np.pad(S_dB, ((0, 0), (0, 2881 - S_dB.shape[1])), 'constant')
    S_dB = S_dB.reshape(-1, 1)
    scaler = joblib.load('./data/models/minmax_scaler.save')
    normalized_melspectrogram = scaler.transform(S_dB.reshape(1,-1))
    test_mels = np.reshape(normalized_melspectrogram,(1,128, -1,1))
    os.remove(wname)
    return test_mels

st.write("""# Music Genre Classifier""")

os.makedirs('./data/test/', exist_ok=True)
link = st.text_input("Paste the link for youtube song \n Example: https://youtu.be/_Yhyp-_hX2s ")
if len(link) >2:
    yt_download_fname = './data/test/test.wav'
    download_clip(link, yt_download_fname)
    while not os.path.isfile(yt_download_fname):
        with st.spinner('File Downloading...'):
            time.sleep(5)
    st.success('Done!')
    X_test = generate_mels(yt_download_fname)
    class_labels = ['Electronic', 'acoustic', 'classical', 'country', 'dance', 'hip-hop', 'jazz',
                    'metal', 'reggae', 'rnb', 'rock']

    cnn_model = create_model( model_type = 'CNN', compile_model=False)
    crnn_model = create_model( model_type = 'CRNN', compile_model=False)
    cnn_model.load_weights('./data/models/CNN/')
    crnn_model.load_weights('./data/models/CRNN/')
    pred_cnn = np.argmax( cnn_model.predict(X_test), axis=1)
    pred_crnn = np.argmax( crnn_model.predict(X_test), axis=1)
    st.write(f"### CNN Model Genre Prediction: ", class_labels[pred_cnn[0]])
    st.write(f"### CRNN Model Genre Prediction: ",class_labels[pred_crnn[0]])
    os.remove(yt_download_fname)