import nltk
nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()
from flask import Flask, jsonify, request
from flask_cors import CORS
import numpy
import tflearn
import tensorflow as tf
from deep_translator import GoogleTranslator
import random #borrar


import json
with open('C:/Users/User/OneDrive/Documentos/Sistemas expertos/chat/my-venv/Lib/site-packages/emociones.json',  encoding="utf8") as file: 
    data = json.load(file)

words = []
labels = []
docs_x = []
docs_y = []



for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs_x.append(wrds)
        docs_y.append(intent["tag"])

    if intent["tag"] not in labels:
        labels.append(intent['tag'])
        


words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words))) #la función 

labels = sorted(labels)
training = []
output = []
out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)

training = numpy.array(training)
output = numpy.array(output)

graph = tf.Graph() #PROBAR SI DA ERROR

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 20)
net = tflearn.fully_connected(net, 20)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=300, batch_size=20, show_metric=True)
model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)

app = Flask(__name__)
CORS(app)
@app.route('/text',methods=['POST'])

def chat():
    dato = request.json #se obtiene la información ingresada desde postman
    print(dato)    
    traductor = GoogleTranslator(source='es', target='en')
    resultado = traductor.translate(dato.get('humanQuestion'))


    results = model.predict([bag_of_words(resultado, words)])

    results_index = numpy.argmax(results)
    tag = labels[results_index]
    print(tag)

    for tg in data["intents"]:
        if tg['tag'] == tag:
            responses = tg['responses']

    ptraducida = random.choice(responses)#borrar
    traductor2 = GoogleTranslator(source='en', target='es')
    resultado2 = traductor2.translate(tag)
    
    return jsonify({'BotResponse': resultado2})

if __name__ == '__main__':
    app.run()
