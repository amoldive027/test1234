import sys
import pyaudio
import numpy as np
import matplotlib.mlab as mlab
from queue import Queue
from pydub import AudioSegment
from pydub.playback import play
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import Dense, Activation, Dropout, Input, TimeDistributed, GRU, Conv1D, BatchNormalization


def model(input_shape):
    """
    Function creating the model's graph in Keras.

    Argument:
    input_shape -- shape of the model's input data (using Keras conventions)

    Returns:
    model -- Keras model instance
    """
    X_input = Input(shape=input_shape)
    # Step 1: CONV layer
    X = Conv1D(filters=256, kernel_size=15, strides=4)(X_input)
    X = BatchNormalization()(X)
    X = Activation("relu")(X)
    X = Dropout(rate=0.8)(X)
    # Step 2: First GRU Layer
    X = GRU(units=128, return_sequences=True, reset_after=False)(X)
    X = Dropout(rate=0.8)(X)
    X = BatchNormalization()(X)
    # Step 3: Second GRU Layer
    X = GRU(units=128, return_sequences=True, reset_after=False)(X)
    X = Dropout(rate=0.8)(X)
    X = BatchNormalization()(X)
    X = Dropout(rate=0.8)(X)
    # Step 4: Time-distributed dense layer (see given code in instructions)
    X = TimeDistributed(Dense(1, activation="sigmoid"))(X)
    model = Model(inputs=X_input, outputs=X)
    return model


def detect_triggerword_spectrum(x, model):
    """
    Function to predict the location of the trigger word.

    Argument:
    x -- spectrum of shape (freqs, Tx)
    i.e. (Number of frequencies, The number time steps)

    Returns:
    predictions -- flattened numpy array to shape (number of output time steps)
    """
    x = x.swapaxes(0, 1)
    x = np.expand_dims(x, axis=0)
    predictions = model.predict(x)
    return predictions.reshape(-1)


def has_new_triggerword(predictions, chunk_duration, feed_duration, threshold=0.5):
    """
    Function to detect new trigger word in the latest chunk of input audio.
    It is looking for the rising edge of the predictions data belongs to the
    last/latest chunk.

    Argument:
    predictions -- predicted labels from model
    chunk_duration -- time in second of a chunk
    feed_duration -- time in second of the input to model
    threshold -- threshold for probability above a certain to be considered positive

    Returns:
    True if new trigger word detected in the latest chunk
    """
    predictions = predictions > threshold
    chunk_predictions_samples = int(len(predictions) * chunk_duration / feed_duration)
    chunk_predictions = predictions[-chunk_predictions_samples:]
    level = chunk_predictions[0]
    for pred in chunk_predictions:
        if pred > level:
            return True
        else:
            level = pred
    return False


def get_spectrogram(data):
    """
    Function to compute a spectrogram.

    Argument:
    predictions -- one channel / dual channel audio data as numpy array

    Returns:
    pxx -- spectrogram, 2-D array, columns are the periodograms of successive segments.
    """
    nfft = 200  # Length of each window segment
    fs = 8000  # Sampling frequencies
    noverlap = 120  # Overlap between windows
    nchannels = data.ndim
    if nchannels == 1:
        pxx, _, _ = mlab.specgram(data, nfft, fs, noverlap=noverlap)
    elif nchannels == 2:
        pxx, _, _ = mlab.specgram(data[:, 0], nfft, fs, noverlap=noverlap)
    return pxx


def get_audio_input_stream(callback, chunk_duration=0.5, fs=44100):
    chunk_samples = int(fs * chunk_duration)  # Each read length in number of samples.
    return pyaudio.PyAudio().open(format=pyaudio.paInt16,
                                  channels=1,
                                  rate=fs,
                                  input=True,
                                  frames_per_buffer=chunk_samples,
                                  input_device_index=0,
                                  stream_callback=callback)


def callback(in_data, frame_count, time_info, status):
    global run, data, silence_threshold
    data0 = np.frombuffer(in_data, dtype='int16')
    if np.abs(data0).mean() < silence_threshold:
        sys.stdout.write('-')
        return in_data, pyaudio.paContinue
    else:
        sys.stdout.write('.')
    data = np.append(data, data0)
    if len(data) > feed_samples:
        data = data[-feed_samples:]
        q.put(data)  # Process data async by sending a queue.
    return in_data, pyaudio.paContinue


def trigger_action():
    # When the trigger word is said, do an action
    sys.stdout.write('1')
    chime = AudioSegment.from_wav("audios/chime.wav")
    play(chime)


Tx = 5511  # The number of time steps input to the model from the spectrogram
n_freq = 101  # Number of frequencies input to the model at each time step of the spectrogram
chunk_duration = 0.5  # Each read length in seconds from mic.
fs = 44100  # sampling rate for mic
feed_duration = 10
silence_threshold = 100
# Each model input data duration in seconds, need to be an integer numbers of chunk_duration
feed_samples = int(fs * feed_duration)
assert feed_duration / chunk_duration == int(feed_duration / chunk_duration)
q = Queue()  # Queue to communiate between the audio callback and main thread
data = np.zeros(feed_samples, dtype='int16')  # Data buffer for the input wavform
run = True

model = model(input_shape=(Tx, n_freq))
# opt = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, decay=0.01)
opt = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)

model.compile(loss='binary_crossentropy', optimizer=opt, metrics=["accuracy"])
model.load_weights('models/tr_model.h5')

stream = get_audio_input_stream(callback, chunk_duration=chunk_duration, fs=fs)
stream.start_stream()
try:
    while run:
        data = q.get()
        preds = detect_triggerword_spectrum(get_spectrogram(data), model)
        new_trigger = has_new_triggerword(preds, chunk_duration, feed_duration)
        if new_trigger:
            trigger_action()
except(KeyboardInterrupt, SystemExit):
    # for KeyboardInterrupt use Ctrl+F2 in Pycharm or Ctrl+C in normal Python
    run = False
finally:
    stream.stop_stream()
    stream.close()
