import emoji
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import tensorflow as tf
import pickle
from sklearn.preprocessing import LabelEncoder
import logging
logging.getLogger('tensorflow').disabled = True

stop_words = set(stopwords.words("english"))

# load Emotion Recognition From English text tensorflow model

model = tf.keras.models.load_model(
    'text/Emotion Recognition 2 LSTM.h5')

# load tokenizer
with open('text/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

with open('text/label_encoder.sav', 'rb') as handle:
    le = pickle.load(handle)


emoji.UNICODE_EMOJI = emoji.EMOJI_DATA


def lemmatization(text):
    lemmatizer = WordNetLemmatizer()

    text = text.split()

    text = [lemmatizer.lemmatize(y) for y in text]

    return " " .join(text)


def remove_stop_words(text):

    Text = [i for i in str(text).split() if i not in stop_words]
    return " ".join(Text)


def Removing_numbers(text):
    text = ''.join([i for i in text if not i.isdigit()])
    return text


def lower_case(text):

    text = text.split()

    text = [y.lower() for y in text]

    return " " .join(text)


def Removing_punctuations(text):
    # Remove punctuations
    text = re.sub('[%s]' % re.escape(
        """!"#$%&'()*+,،-./:;<=>؟?@[\]^_`{|}~"""), ' ', text)
    text = text.replace('؛', "", )

    # remove extra whitespace
    text = re.sub('\s+', ' ', text)
    text = " ".join(text.split())
    return text.strip()


def Removing_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


def remove_small_sentences(df):
    for i in range(len(df)):
        if len(df.text.iloc[i].split()) < 3:
            df.text.iloc[i] = np.nan


def normalize_text(df):
    df.Text = df.Text.apply(lambda text: lower_case(text))
    df.Text = df.Text.apply(lambda text: remove_stop_words(text))
    df.Text = df.Text.apply(lambda text: Removing_numbers(text))
    df.Text = df.Text.apply(lambda text: Removing_punctuations(text))
    df.Text = df.Text.apply(lambda text: Removing_urls(text))
    df.Text = df.Text.apply(lambda text: lemmatization(text))
    return df


def normalized_sentence(sentence):
    sentence = lower_case(sentence)
    sentence = remove_stop_words(sentence)
    sentence = Removing_numbers(sentence)
    sentence = Removing_punctuations(sentence)
    sentence = Removing_urls(sentence)
    sentence = lemmatization(sentence)
    return sentence


def pred(sentence):
    sentence = normalized_sentence(sentence)
    sentence = tokenizer.texts_to_sequences([sentence])
    sentence = pad_sequences(sentence, maxlen=229, truncating='pre')
    result = le.inverse_transform(
        np.argmax(model.predict(sentence), axis=-1))[0]
    proba = np.max(model.predict(sentence))

    # joy, sadness,anger, fear,love,surprise
    # replace with sad angry happy fear surprise
    if result == 'joy' or result == 'love':
        result = "happy"
    elif result == 'sadness':
        result = "sad"
    elif result == 'anger':
        result = "angry"

    return result
