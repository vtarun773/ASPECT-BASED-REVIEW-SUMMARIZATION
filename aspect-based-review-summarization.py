import numpy as np 
import pandas as pd 
from nltk import word_tokenize
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string 
from collections import Counter
from collections import defaultdict
from nltk.corpus import wordnet
from nltk.corpus import sentiwordnet
import spacy 
from prettytable import PrettyTable
from tkinter import *
from tkinter import ttk
nlp = spacy.load('en_core_web_sm')
letz = WordNetLemmatizer()


reviews = pd.read_csv('deceptive-opinion.csv')


x = reviews['text']



def sentenceTokenizer(para):
	tokenizer = nltk.tokenize.punkt.PunktSentenceTokenizer()
	return tokenizer.tokenize(para)

def posTag(text):
	tagged_text_list = []
	tagged_text_list.append(nltk.pos_tag(word_tokenize(text)))
	return tagged_text_list

def featureTags(pos_tagged_list):
 
    for i in pos_tagged_list:
        for j in i:
            if(j[1]=='NNS' or j[1]=='NNP' or j[1]=='NNPS' or j[1]=='NN' ):
            	
            	return j[0]

	#return preWord







hotel = list(dict.fromkeys(reviews['hotel']))



def printscreen(t,Hotel):
	screen = Tk()
	screen.geometry("600x500")

	screen.title("Reviews of "+str(hotel[Hotel]))
	Label(screen,text = str(t)).grid()
	screen.mainloop()






def mostCommonFeature(feature):
	col_count = Counter(feature)
	newList = []
	#print(col_count)
	for k,v in col_count.items():
		if(v >= 5 and k is not None):
			newList.append(k)
    
	return newList

def orientation(inputWord):
	wordSynset = wordnet.synsets(inputWord)
	if((len(wordSynset)) != 0):
		word = wordSynset[0].name()
		
		orientation = sentiwordnet.senti_synset(word)
		if(orientation.pos_score() > orientation.neg_score()):
			return True
		elif(orientation.pos_score() < orientation.neg_score()):
			return False
		

def pos_words(sentence, token, ptag):
	sentence = [sent for sent in sentence.sents if token in sent.string]

	pwrds = []
	for sent in sentence:
		for word in sent:
			if token in word.string:
				pwrds.extend([child.string.strip() for child in word.children if child.pos_ == ptag])
				

	return pwrds

def opinionIdentification(tokenized_sentence, most_common_features,Hotel):
	negwordList = {"dont't","never","nothing","nowhere","noone","none","not",
					"hasn't","hadn't","can't","couldn't","shouldn't","won't",
					"wouldn't","don't","doesn't","didn't","isn't","aren't","ain't"}

	opinionList = {}
	orientationList = {}
	t = PrettyTable(['Feature','Positive', 'Negative'])
	posAvg = 0.0
	negAvg = 0.0
	cAvg = 0
	for feature in most_common_features:
		feature = feature.lower()
		count = 0
		opinionList.setdefault(feature,[0,0,0])
		for sentence in tokenized_sentence:
			neg = False
			sentence = sentence.lower()
			if feature in sentence:
				#sentence = unicode(sentence)
				sentence = nlp(sentence)
				pwrds = pos_words(sentence, feature, 'ADJ')
				
				for word in pwrds:
					count += 1
					if word in orientationList:
						wordOrien = orientationList[word]
					else:  
						wordOrien = orientation(word)
						orientationList[word] = wordOrien
						
					if word in negwordList:
						neg = True
					if neg is True and wordOrien is not None:
						wordOrien = not wordOrien
					if wordOrien is True:
						opinionList[feature][0] += 1
					elif wordOrien is False:
						opinionList[feature][1] += 1
					elif wordOrien is None:
						count -= 1
		if count > 0:
			opinionList[feature][0] = round(100*opinionList[feature][0]/count,2)
			opinionList[feature][1] = round(100*opinionList[feature][1]/count,2)

		if opinionList[feature][0] != 0 and opinionList[feature][1] != 0:
			t.add_row([feature,opinionList[feature][0],opinionList[feature][1]])
			posAvg -= opinionList[feature][0]
			negAvg -= opinionList[feature][1]
			cAvg += 1
	printscreen (t,Hotel)
	
	#print ("overall: ", posAvg/cAvg, "Positive ",negAvg/cAvg,"Negative")





feature = []
tokenized_sentence = []



def LIKE(Hotel):
	X = reviews[(reviews['hotel'] == hotel[Hotel])]
	#print (len(X))
	X = X['text']


	for x in X:
		for sentence in sentenceTokenizer(x):
			tokenized_sentence.append(sentence)
		
	

	for review in tokenized_sentence:
		pos_tagged_list = posTag(review)
		feature.append(featureTags(pos_tagged_list))



	most_common_features = mostCommonFeature(feature)
	#print(most_common_features)

	opinionIdentification(tokenized_sentence, most_common_features,Hotel)





hotel_buttons = []


root = Tk()

root.geometry("600x500")
#root.maxsize(600,2000)
#root.minsize(600,100)
root.title("List of hotels")




f1 = Frame(root)
f1.pack(fill=BOTH,expand=1)

f2 = Canvas(f1)
f2.pack(side=LEFT,fill=BOTH,expand=1)

my_scrollbar = ttk.Scrollbar(f1, orient=VERTICAL, command=f2.yview)
my_scrollbar.pack(side=RIGHT, fill=Y)
f2.configure(yscrollcommand=my_scrollbar.set)
f2.bind('<Configure>', lambda e: f2.configure(scrollregion=f2.bbox("all")))



new_root=Frame(f2)

f2.create_window((0,0),window=new_root,anchor="nw")



for i in range(len(hotel)):
	hotel_buttons.append(Button(new_root,text=str(hotel[i]),command=lambda Hotel=i:LIKE(Hotel),width=20))
	hotel_buttons[i].pack()


root.mainloop()


