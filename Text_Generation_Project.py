#!/usr/bin/env python
# coding: utf-8

# In[1]:


text = open('book.txt','r',encoding='utf8').read()
text = text.lower()


# In[2]:


# split whole text in list of sentences
sentences = text.split('\n')


# In[3]:


import tensorflow as tf
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


# In[4]:


# now we are going to tokenize our text data i.e sentences which mean we are going to assign every word a unique integer
# so to do that first we have to declare object of tokenizer

# their are several argument we cann pass in tokenizer but we only make oov_token = <UNK> (unknown)
# we use this as may be their are several word in data which are not present in our vocbulary(dictionary words from text)
# so if we didn't pass it skip those word as its default value is NONE so we initialize it as unknown

tokenizer = Tokenizer(oov_token='<UNK>')


# In[5]:


tokenizer.fit_on_texts(sentences)


# In[6]:


vocab_size = len(tokenizer.word_index)+1
vocab_size


# In[7]:


# now we will convert our sentences into integr sequences i.e for every sentence it going to replace word in sentence with corresponding 
# integer value that we get through fit_on_text function

sequences = tokenizer.texts_to_sequences(sentences)


# In[8]:


# now we convert our sequences into n-gram sequences
# for ex we have sequence as [1,3,21,5,6]
# so with this we can make more sequences as [1,3],[1,3,21],[1,3,21,5],[1,3,21,5,6] (we keep min two word as one sequences)
input_sequences = []
for sq in sequences:
  for i in range(1,len(sq)):
    n_gram_sequence = sq[:i+1]
    input_sequences.append(n_gram_sequence)

# lets print first two sequences
print(input_sequences[0])
print(input_sequences[1])



# In[9]:


# now we are going to pad our sequence because a we are going to use RNN in our project and it require input of same length
# so what padding does it make make all sequence of same length by adding required no. of 0 in front of orginal seuence

# for example we have [124,470] and in our data we have max length of sequence is 5 so it will convert it into [0,0,0,146,4790]

# so first we find maxlength
max_length = max([len(sq) for sq in input_sequences])
max_length


# In[10]:


padded_sequences = pad_sequences(input_sequences,max_length)
print(padded_sequences[0])
print(padded_sequences[1])


# In[11]:


# lets covert or padded sequences into numpy arry for easy computation
import numpy as np
padded_sequences = np.array(padded_sequences)


# In[12]:


# now we are going to make our input output pair or data 
# so for that what we do as we have have sequences of len 20 we kept 19 as input data and the last one as output data
# basic idea is we have some text sentence and we can predict what is the next word

x= padded_sequences[:,:-1] # i.e all row and column(expect last one)
labels=padded_sequences[:,-1]


# In[13]:


# since in this we are going to predict next word from some given word so basicall it is a kind of classification problem
# so we convert our label data into one hot encoded format as it is the requirment of neural networks

y=tf.keras.utils.to_categorical(labels, num_classes=vocab_size)


# In[14]:


x.shape


# In[15]:


y.shape


# In[17]:


# import necessary library
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding,Bidirectional,LSTM,Dense
from tensorflow.keras.optimizers import Adam


# In[20]:


# define and compile the model
model = Sequential()
model.add(Embedding(vocab_size,100,input_length=max_length-1))
model.add(Bidirectional(LSTM(256)))
model.add(Dense(vocab_size,activation='softmax'))
adam = Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy',optimizer=adam,metrics = ['acc'])
model.summary()


# In[21]:


# initialize the callback function to early stop the training of data if their is not atleast 1% improvment in accuracy
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor ='acc',min_delta=0.01) 


# In[22]:


model.fit(x,y,epochs=50,verbose=1,batch_size=512,callbacks=[es])


# In[27]:


# time for story telling or prediction
seed_text = " I warned you about this thing and"
next_words = 50
for _ in range(next_words):
  sequence = tokenizer.texts_to_sequences([seed_text])
  padded = pad_sequences(sequence,maxlen=max_length-1)
  predicted = np.argmax(model.predict(padded,verbose=0),axis=1)
  output_word =''
  for word,index in tokenizer.word_index.items():
    if index==predicted:
      output_word = word
      break
  seed_text += ' ' + output_word
print(seed_text)


# In[43]:


# lets look at how loss and accuracy change while training
"""
import matplotlib.pyplot as plt
history = model.history
print(history.history)
#ac = history.history['acc']
loss = history.history['loss']

epoch = range(len(ac))

#plt.plot(epoch,ac,'b',label = 'Training Accuracy')
#plt.title('Training Accuracy')

#plt.figure()

plt.plot(epoch,loss,'b',label = 'Training Loss')
plt.title('Training Loss')
plt.legend()

plt.show()
"""


# In[ ]:




