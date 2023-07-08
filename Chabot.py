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
#Carga de la informacion contenida en el json.
with open('C:/Users/User/OneDrive/Documentos/Sistemas expertos/chat/my-venv/Lib/site-packages/emociones.json',  encoding="utf8") as file: 
    data = json.load(file)
    
#Declaración de variables necesarias. 
words = []
labels = []
docs_x = []
docs_y = []


#Proceso de tratamiento de la data antes de entrenar
for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern) #Función para para separar las frases en palabras y contenterlas en un array, ejemplo (Estoy jugando futbol), la salida de la función sería ['Estoy', 'jugando', 'futbol.'] 
        words.extend(wrds) #Se agrega al final de la lista
        #Se agrega al array separados, para ser utilizados en el entrenamiento de la data y la etiqueta.
        docs_x.append(wrds)
        docs_y.append(intent["tag"])
#Si la etiqueta no está en el label, la agrega en él.
    if intent["tag"] not in labels:
        labels.append(intent['tag'])
        

#Se vuelve a procesar las palabras, toquenizadas, es decir, 
#en esta parte, se vuelven las palabras a su verbo infitivo, y se vuelven minisculas
#A su vez se descartan las palabra si es igual ?
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
#Ordena las palabras y ahora en el array no van a haber palabras repetidas.
words = sorted(list(set(words))) 

#Ordena las etiquetas.
labels = sorted(labels)
#Se crean los arreglos y las salidas esperadas.
training = []
output = []
#Se declara un salida deseada del tamaño de las etiquetas en 0
out_empty = [0 for _ in range(len(labels))]

#Por ultimo, preparar los datos para entrenamiento y las salidas de la siguiente forma

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds: #Agrega un 1 se la la palabra se encuentra en las palabras toquenizadas anteriormente, con respecto al array words que además de la toquenizacion, se hacen infinitivas las palabras. 
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag) #(se agrega la data entrenada. )
    output.append(output_row)
#Se hace los arreglos multidimensionales para entrenar con la libreria Tflearn de tensorFlow
training = numpy.array(training)
output = numpy.array(output)
Se utiliza un graph para separar si estuvisesemos entrenando multiples
graph = tf.Graph() 

net = tflearn.input_data(shape=[None, len(training[0])]) #Una capa de entrada de datos con el tamaño de la data para entranamiento
net = tflearn.fully_connected(net, 20) #Se define dos capas una conectada a la anterior con 20 neuronas.
net = tflearn.fully_connected(net, 20)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax") #Softmax se utiliza para que nos devuelva la probabilidad, es utilizado en la capa de salida, y la suma de todas
net = tflearn.regression(net) #Muestra la perdida que va duranete el entrenamiento

model = tflearn.DNN(net) #El modelo de red neuronal, Aprendazaje profundo

model.fit(training, output, n_epoch=300, batch_size=20, show_metric=True)
model.save("model.tflearn")

#Le metodo de la bolsas de palabras para encontrar para encontrar el valor
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
