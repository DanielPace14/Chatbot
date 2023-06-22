import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
from keras.models import load_model
import json
import random
import os
import datetime
import time
import requests

lemmatizer = WordNetLemmatizer()

intents = json.loads(open(f'{os.getcwd()}\\intents.json').read())
words = pickle.load(open(f'{os.getcwd()}\\words.pkl','rb'))
classes = pickle.load(open(f'{os.getcwd()}\\classes.pkl','rb'))
model = load_model(f'{os.getcwd()}\\chatbot_model.h5')

def clean_up_sentence(sentence):
    
    # tokenize input pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    
    return sentence_words
    

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
                    
    return(np.array(bag))


def predict_class(sentence, model):
    
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if tag=='datetime':
            return(time.strftime("%A")+"\n"+time.strftime("%d %B %Y")+"\n"
                   +time.strftime("%H:%M:%S"))
        if tag=='weather':
            api_key = "5a2c5600149ab06c6038339f7af691f7"
            base_url = "http://api.openweathermap.org/data/2.5/weather?"
            city = "Montreal"
            request = base_url+ "appid="+api_key+"&q="+city
            response = requests.get(request)
            values = response.json()
            main_data = values["main"]
            temp = str(round(main_data["temp"]-273.2))
            hum = str(main_data["humidity"])
            description = values['weather'][0]['description']
            return("Temperature: "+temp+u'\N{DEGREE SIGN}'+" C \n"+
                   'Humidity: '+hum+"\n"+
                   "Weather Description: "+description)
        
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(text):
    ints = predict_class(text, model)
    res = getResponse(ints, intents)
    return res
if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = chatbot_response(sentence)
        print(resp)
# import tkinter
# from tkinter import *

# # send function: add entry to chat window and get chatbot response
# def send():
#     # get written message and save to variable
#     msg = EntryBox.get("1.0",'end-1c').strip()
#     # remove message from entry box
#     EntryBox.delete("0.0",END)
    
#     if msg == "Message":
#         # if the user clicks send before entering their own message, "Message" gets inserted again
#         # no prediction/response
#         EntryBox.insert(END, "Message")
#         pass
#         # if user clicks send and proper entry
#     elif msg != '':
#         # activate chat window and insert message
#         ChatLog.config(state=NORMAL)
#         ChatLog.insert(END, "You: " + msg + '\n\n')
#         ChatLog.config(foreground="black", font=("Verdana", 12 ))
        
#         # insert bot response to chat window
#         res = chatbot_response(msg)
#         ChatLog.insert(END, "DanBot:\n" + res + '\n\n')
        
#         # make window read-only again
#         ChatLog.config(state=DISABLED)
#         ChatLog.yview(END)

# def clear_search(event):
#     EntryBox.delete("0.0",END)
#     EntryBox.config(foreground="black", font=("Verdana", 12))
    
# base = Tk()
# base.title("DanBot")
# base.geometry("400x500")
# text=Text(base, width = 50, height = 50, 
#           wrap = WORD, padx = 10, pady = 10)
# text.pack()
# base.resizable(width=FALSE, height=FALSE)

# # create chat window
# ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",wrap = WORD, padx = 10, pady = 10)

# # initial greeting in chat window
# ChatLog.config(foreground="black", font=("Verdana", 12 ))
# ChatLog.insert(END, "DanBot: Hello, I can help with... \n\t- Skills \n\t- Education \n\t- Time and Weather\n\n")
# # disable window = read-only
# ChatLog.config(state=DISABLED)

# # bind scrollbar to ChatLog window
# scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
# ChatLog['yscrollcommand'] = scrollbar.set

# # create Button to send message
# SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="9", height=5,
#                     bd=0, bg="blue", activebackground="gold",fg='#ffffff',
#                     command= send )

# # create the box to enter message
# EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial",wrap = WORD, padx = 10, pady = 10)
# EntryBox.pack()
# # display a grey text in EntryBox that's removed upon clicking or tab focus
# EntryBox.insert(END, "Message")
# EntryBox.config(foreground="grey", font=("Verdana", 12))
# EntryBox.bind("<Button-1>", clear_search)
# EntryBox.bind("<FocusIn>", clear_search) 

# # place components at given coordinates in window (x=0 y=0 top left corner)
# scrollbar.place(x=376,y=6, height=386)
# ChatLog.place(x=6,y=6, height=386, width=370)
# EntryBox.place(x=6, y=401, height=90, width=265)
# SendButton.place(x=282, y=401, height=90)

# base.mainloop()
