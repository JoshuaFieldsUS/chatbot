import random
import json
import pickle
import numpy as np
import vlc
import pyaudio
import wave
import time
from ibm_watson import SpeechToTextV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

apikey_S2T = 'ynt5hi60AsQmUnm0UFqyfjhGtuarGetL95yyzBF3L8vm'
url_S2T = 'https://api.eu-gb.speech-to-text.watson.cloud.ibm.com/instances/e4884505-38fd-4244-a6be-6225e99b4d44'

#Setup Service
authenticator = IAMAuthenticator(apikey_S2T)
stt = SpeechToTextV1(authenticator=authenticator)
stt.set_service_url(url_S2T)



url = 'https://api.eu-gb.text-to-speech.watson.cloud.ibm.com/instances/777c90c6-6be0-4f52-8410-f6f7d9da2716'
apikey = '7BhfEw5-MjLiHzY2IfYkqm0W0NKlXvbCv8-SOmyffLIr'

from ibm_watson import TextToSpeechV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator

#Audio service set up
authenticator = IAMAuthenticator(apikey)

#New TTS Service
tts = TextToSpeechV1(authenticator)

#Set Sevice URL
tts.set_service_url(url)










import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
        return return_list

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("I noticed you're feeling something, want to talk?")

while True:
    # Audio Setup
    time.sleep(4)
    audio = pyaudio.PyAudio()

    stream = audio.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, frames_per_buffer=1024)

    frames = []

    t_end = time.time() + 6
    #Record Voice
    try:
        print('Time to speak!')
        while time.time() < t_end:
            data = stream.read(1024)
            frames.append(data)
    except:
        pass

    stream.stop_stream()
    stream.close()
    audio.terminate()

    sound_file = wave.open("myrecording.wav", "wb")
    sound_file.setnchannels(1)
    sound_file.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
    sound_file.setframerate(44100)
    sound_file.writeframes(b''.join(frames))
    sound_file.close()

    #Speech to Text
    with open('myrecording.wav', 'rb') as f:
        S2T_res = stt.recognize(audio=f, content_type='audio/wav', model='en-US_NarrowbandModel').get_result()
        S2T_res

        text = S2T_res['results'][0]['alternatives'][0]['transcript']
        text
        confidence = S2T_res['results'][0]['alternatives'][0]['confidence']
        print(text)



    message = text
    ints = predict_class(message)
    res = get_response(ints, intents)
    print(res)
    with open('./Vocal_Response.mp3', 'wb') as audio_file:
        vocal_res = tts.synthesize(res, accept='audio/mp3', voice='en-US_AllisonV3Voice').get_result()
        audio_file.write(vocal_res.content)
    p = vlc.MediaPlayer("C:/Users/Joshu/Desktop/chatbot/Vocal_Response.mp3")
    p.play()

