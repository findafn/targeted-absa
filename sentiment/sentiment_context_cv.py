
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd
import csv
import re
import codecs
import json
import requests
import pickle
from gensim.models import Word2Vec

import tensorflow as tf
from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score, auc
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_validate
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

import keras
from keras import backend as k
from keras_pos_embd import PositionEmbedding
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.engine.topology import Layer
from keras.constraints import max_norm
from keras.models import Sequential, Model, load_model
from keras.optimizers import Adam, Adagrad
from keras.layers import GRU, LSTM, Bidirectional, GlobalMaxPool1D, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.layers import Dense, Dropout, Input, Embedding, TimeDistributed
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences

from sklearn.model_selection import ParameterGrid


# In[2]:

with open('../aspect_detection/w2v_path.txt') as file:
    word2vec_path = file.readlines()[0]
    
w2v = Word2Vec.load(word2vec_path)


# In[3]:

config = tf.ConfigProto()
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.3

k.tensorflow_backend.set_session(tf.Session(config=config))


# In[7]:

ASPECT_LIST = [
    'others',
    'machine',
    'part',
    'price',
    'service',
    'fuel'
    ]

max_length = 50
    
with open('../../../targeted-absa/data/entity_train.json', 'r') as f:
    entity_train = json.load(f)
with open('../../../targeted-absa/data/entity_test.json', 'r') as f:
    entity_test = json.load(f)
    
with open('../../../targeted-absa/resource/postag_train_auto.json') as f:
    pos_train_json = json.load(f)
with open('../../../targeted-absa/resource/postag_test_auto.json') as f:
    pos_test_json = json.load(f)
with open('../../../targeted-absa/resource/pos_dict.json') as f:
    pos_dict = json.load(f)
    
pos_size = len(pos_dict) + 2
    
# with open('../../../targeted-absa/aspect/model/April-18-2019_17-57-48/pred.txt', 'r', encoding='utf-8') as f:
#     pred_label = f.read().splitlines()
    
# aspect_pred_label = list()
# for label in pred_label:
#     temp = list()
#     for char in label:
#         if char == '0' or char == '1':
#             temp.append(int(char))
#     aspect_pred_label.append(temp)
    
# aspect_pred_label = np.array(aspect_pred_label)
    
def read_data_for_aspect(entity_file):
    review = list()
    for data in entity_file:
        for _ in data['info']:
            clean = re.sub(r"[,.;@?!#&$]+\ *", " ", data['sentence'])
            review.append(clean.lower())
    return review

def read_data_for_sentiment(entity_file):
    review = list()
    for datum in entity_file:
        for info in datum['info']:
            for aspect in info['aspect']:
                clean = re.sub(r"[,.;@?!#&$]+\ *", " ", datum['sentence'])
                review.append(clean.lower())
    return review

def read_data_for_sentiment_from_prediction(rev_aspect, pred):
    review = list()
    for i, datum in enumerate(pred):
        for aspect in datum:
            if aspect == 1:
                review.append(rev_aspect[i])
    return review

def read_sentiment(entity_test):
    label = list()
    aspects = list()
    
    for i, datum in enumerate(entity_test):        
        for info in datum['info']:
            for aspect in info['aspect']:
                if aspect.split('|')[1] == 'negative':
                    label.append(0)
                elif aspect.split('|')[1] == 'positive':
                    label.append(1)
                aspects.append(aspect.split('|')[0])
    label = to_categorical(label, num_classes=2)
                
#     for i, lab in enumerate(label):
#         print(i)
#         print(review[i])
#         print(aspects[i], lab)
    return label

def get_context(review_a, entity_file):
    list_left = list()
    list_right = list()
    list_target = list()

    idx = 0
    for sentence in entity_file:            
        for ent in sentence['info']:
            left = list()
            right = list()
            target = list()
            split = review_a[idx].split()
            if ent['name'] != None:                
                entity = ent['name'].lower()
                entity = re.sub('ku', '', entity)
                entity = re.sub('-nya', '', entity)
                entity = re.sub('nya', '', entity)
                e_split = entity.split()
                e_first = e_split[0]
                idx += 1

                for token in split:
                    if e_first in token:
                        loc = split.index(token)
                        break
                for j in range(0,loc):
                    left.append(split[j])
                for j in range(len(e_split)):
                    target.append(e_split[j])
                for j in range(loc+len(e_split), len(split)):
                    right.append(split[j])
                
                for _ in ent['aspect']:
                    list_left.append(' '.join(left))
                    list_right.append(' '.join(right))
                    list_target.append(' '.join(target))
            else:
                split = review_a[idx].split()
                idx += 1    
                left = split
                for _ in ent['aspect']:
                    list_left.append(' '.join(left))
                    list_right.append(right)
                    list_target.append(target)

    return list_left, list_target, list_right

def get_context_from_prediction(review_a, entity_file, pred):
# cari bentuk masukan sesuai aspek prediksi model aspek
    list_left = list()
    list_right = list()
    list_target = list()

    idx = 0
    for sentence in entity_file:            
        for ent in sentence['info']:
            left = list()
            right = list()
            target = list()
            split = review_a[idx].split()
            if ent['name'] != None:                
                entity = ent['name'].lower()
                entity = re.sub('ku', '', entity)
                entity = re.sub('-nya', '', entity)
                entity = re.sub('nya', '', entity)
                e_split = entity.split()
                e_first = e_split[0]

                for token in split:
                    if e_first in token:
                        loc = split.index(token)
                        break
                for j in range(0,loc):
                    left.append(split[j])
                for j in range(len(e_split)):
                    target.append(e_split[j])
                for j in range(loc+len(e_split), len(split)):
                    right.append(split[j])
                
                for pr in pred[idx]:
                    if pr == 1:
                        list_left.append(' '.join(left))
                        list_right.append(' '.join(right))
                        list_target.append(' '.join(target))
                idx += 1
            else:
                split = review_a[idx].split()
                left = split
                for pr in pred[idx]:
                    if pr == 1:
                        list_left.append(' '.join(left))
                        list_right.append(right)
                        list_target.append(target)
                idx += 1    

    return list_left, list_target, list_right

def get_tokenized(review, max_length):
    encoded_data = token.texts_to_sequences(review)
    x_train = pad_sequences(encoded_data, maxlen=max_length, padding='post')
    
    return np.array(x_train)

def get_embedding(w2v, vocab_size):
    words = list(w2v.wv.vocab)
    embeddings_index = dict()

    for f in range(len(words)):
        coefs = w2v[words[f]]
        embeddings_index[words[f]] = coefs

    embedding_matrix = np.zeros((vocab_size, 500))

    for word, i in token.word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
        else:
            embedding_matrix[i] = np.random.rand(500)
            
    return embedding_matrix

def read_pos(json_data, pos_data):
    list_left = list()
    list_right = list()
    list_target = list()

    for i, data in enumerate(json_data):
        split = re.sub(r"[,.;@?!#&$]+\ *", " ", data['sentence'])
        split = split.lower().split()
        idx = 0
        for ent in data['info']:
            left = list()
            right = list()
            target = list()
            if ent['name'] != None:
                entity = ent['name'].lower()
                entity = re.sub('ku', '', entity)
                entity = re.sub('-nya', '', entity)
                entity = re.sub('nya', '', entity)
                e_split = entity.split()
                e_first = e_split[0]
                
                for token in split:
                    if e_first in token:
                        loc = split.index(token)
                        break
                    
                for j in range(0, loc):
                    left.append(pos_dict[pos_data[i]['sentences'][0]['tokens'][j]['pos_tag']])
                    if len(left) == max_length:
                        break  
                left = np.pad(left, (0,max_length-len(left)), 'constant', constant_values=(0))
                
                for j in range(loc,len(e_split)):
                    target.append(pos_dict[pos_data[i]['sentences'][0]['tokens'][j]['pos_tag']])
                    if len(target) == 10:
                        break  
                target = np.pad(target, (0,10-len(target)), 'constant', constant_values=(0))
                
                for j in range(loc+len(e_split), len(pos_data[i]['sentences'][0]['tokens'])):
                    right.append(pos_dict[pos_data[i]['sentences'][0]['tokens'][j]['pos_tag']])
                    if len(right) == max_length:
                        break  
                right = np.pad(right, (0,max_length-len(right)), 'constant', constant_values=(0))

        #         if self.module_name == 'aspect':
        #             if self.use_entity:
        #                 for _ in data['info']:
        #                     pos.append(temp)
        #             else:
        #                 pos.append(temp)
        #         elif self.module_name == 'sentiment':
        #             if self.use_entity:
                for _ in ent['aspect']:
                    list_left.append(left)
                    list_right.append(right)
                    list_target.append(target)

            else:
                for j in range(len(pos_data[i]['sentences'][0]['tokens'])):
                    left.append(pos_dict[pos_data[i]['sentences'][0]['tokens'][j]['pos_tag']])
                    idx += 1
                    if idx == max_length - 1:
                        break  
                left = np.pad(left, (0,max_length-len(left)), 'constant', constant_values=(0))
                target = np.zeros(10, dtype=int)
                right = np.zeros(max_length, dtype=int)
                        
                for _ in ent['aspect']:
                    list_left.append(left)
                    list_right.append(right)
                    list_target.append(target)

    list_left = np.array(list_left)
    list_right = np.array(list_right)
    list_target = np.array(list_target)
    return list_left, list_target, list_right

pos_train_left, pos_train_target, pos_train_right = read_pos(entity_train, pos_train_json)
pos_test_left, pos_test_target, pos_test_right = read_pos(entity_test, pos_test_json)

review_atr = read_data_for_aspect(entity_train)
review_ate = read_data_for_aspect(entity_test)

review_str = read_data_for_sentiment(entity_train)
review_ste = read_data_for_sentiment(entity_test)


left_tr, target_tr, right_tr = get_context(review_atr, entity_train)
left_te, target_te, right_te = get_context(review_ate, entity_test)

# aspect = get_aspect_from_json(json_train)
# aspect_test = get_aspect_from_json(json_test)
# y_train = get_label_from_json(json_train, review)
# y_test = get_label_from_json(json_test, review_test)

y_train = read_sentiment(entity_train)
y_test = read_sentiment(entity_test)

token = Tokenizer()
token.fit_on_texts(review_atr)
vocab_size = len(token.word_index) + 1

train_left = get_tokenized(left_tr, max_length)
train_target = get_tokenized(target_tr, 10)
train_right = get_tokenized(right_tr, max_length)

test_left = get_tokenized(left_te, max_length)
test_target = get_tokenized(target_te, 10)
test_right = get_tokenized(right_te, max_length)

embedding_matrix = get_embedding(w2v, vocab_size)


# In[10]:

def read_aspect(data):
    label = list()

    for i, datum in enumerate(data):
        for info in datum['info']:
            temp = list()
            for aspect in info['aspect']:
                temp.append(aspect.split('|')[0])
            label.append(temp)

    encoded_label = list()
    for aspects in label:
        temp = np.zeros(len(ASPECT_LIST), dtype=int)
        for aspect in aspects:
            for i, asp in enumerate(ASPECT_LIST):
                if asp in aspect:
                    temp[i] = 1
        encoded_label.append(temp)

    print('Label shape  :', np.array(encoded_label).shape)    
    print('Example label:', encoded_label[0])

    return np.array(encoded_label)

def change_aspect_to_one_hot(label, forwhat):
    aspect_train = list()
    for i in range(len(label)):
        for j in range(len(label[i])):
            if label[i][j] == 1:
                temp = list()
                if forwhat == 'target':
                    length = 10
                elif forwhat == 'notarget':
                    length = 50
                elif forwhat == 'all':
                    length = 110
                for _ in range(length):
                    temp.append(j)
                aspect_train.append(temp)
    aspect_train = np.array(aspect_train)
    return aspect_train

def change_to_multilabel(aspects, sentiments):
    multilabel = list()
    idx = 0
    for data in aspects:
        temp = list()
        for aspect in data:
            if aspect == 1:
                if sentiments[idx] == 0:
                    temp.append(2)
                else:
                    temp.append(1)
                idx += 1
            else:
                temp.append(0)
        multilabel.append(temp)

    return np.array(multilabel)


# In[11]:

x_aspect_train = read_aspect(entity_train)
x_aspect_test = read_aspect(entity_test)

aspect_train_target = change_aspect_to_one_hot(x_aspect_train, 'target')
aspect_train = change_aspect_to_one_hot(x_aspect_train, 'all')
aspect_test_target = change_aspect_to_one_hot(x_aspect_test, 'target')
aspect_test = change_aspect_to_one_hot(x_aspect_test, 'all')


# In[18]:

def build_model():
    left_input = Input(shape=(max_length,), dtype='int32', name='left_input')
    x1 = Embedding(
        output_dim=500, 
        input_dim=vocab_size, 
        input_length=max_length,
        weights=[embedding_matrix],
        trainable=True,
    )(left_input)

    pos_left_input = Input(shape=(max_length,), dtype='int32', name='pos_left_input')
    x1p = keras.layers.Lambda(
        k.one_hot, 
        arguments={'num_classes': pos_size}, 
        output_shape=(max_length, pos_size)
    )(pos_left_input)

    target_input = Input(shape=(10,), dtype='int32', name='target_input')
    x2 = Embedding(
        output_dim=500, 
        input_dim=vocab_size, 
        input_length=10,
        weights=[embedding_matrix],
        trainable=True,
    )(target_input)

    pos_target_input = Input(shape=(10,), dtype='int32', name='pos_target_input')
    x2p = keras.layers.Lambda(
        k.one_hot, 
        arguments={'num_classes': pos_size}, 
        output_shape=(10, pos_size)
    )(pos_target_input)

    right_input = Input(shape=(max_length,), dtype='int32', name='right_input')
    x3 = Embedding(
        output_dim=500, 
        input_dim=vocab_size, 
        input_length=max_length,
        weights=[embedding_matrix],
        trainable=True,
    )(right_input)

    pos_right_input = Input(shape=(max_length,), dtype='int32', name='pos_right_input')
    x3p = keras.layers.Lambda(
        k.one_hot, 
        arguments={'num_classes': pos_size}, 
        output_shape=(max_length, pos_size)
    )(pos_right_input)

    aspect_input = Input(shape=(110,), dtype='int32', name='aspect_input')
    x4 = keras.layers.Lambda(
        k.one_hot, 
        arguments={'num_classes': len(ASPECT_LIST)}, 
        output_shape=(110, len(ASPECT_LIST))
    )(aspect_input)

    x = keras.layers.concatenate([x1, x2, x3], axis=1)
    xp = keras.layers.concatenate([x1p, x2p, x3p], axis=1)
    x = keras.layers.concatenate([x, xp, x4])


    x = Bidirectional(LSTM(256, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(2, activation='softmax')(x)

    model = Model([left_input, target_input, right_input, pos_left_input, pos_target_input, pos_right_input, aspect_input], out)

    model.summary()

    model.compile(
        loss='categorical_crossentropy', 
        optimizer='adam', 
        metrics=['acc']
    )
    return model


# In[13]:

checkpoint = ModelCheckpoint(
    'crossval/model_context.h5', 
    monitor='val_acc', 
    verbose=0, 
    save_best_only=True, 
    mode='max', 
    period=1
)


# In[14]:

x_input = [train_left, train_target, train_right, pos_train_left, pos_train_target, pos_train_right, aspect_train]


# In[20]:

kfold = KFold(n_splits=2,random_state=70,shuffle=True)
histos = list()
for train, val in kfold.split(x_input[0]):
    x_fold_train = list()
    x_fold_val = list()
    for i in range(len(x_input)):
        x_fold_train.append(x_input[i][train])
        x_fold_val.append(x_input[i][val])
    y_fold_train, y_fold_val = y_train[train], y_train[val]
    model = build_model()
    hist = model.fit(
        x_fold_train, 
        y_fold_train, 
        batch_size=16, 
        epochs=1, 
        verbose=1,
        validation_data=[x_fold_val,y_fold_val],
        callbacks=[checkpoint],
    )
    histos.append(hist)

val_acc = np.zeros((len(histos), len(histos[0].history['val_acc'])))
for i in range(len(histos)):
    print(histos[i].history['val_acc'])
    val_acc[i]= np.asarray(histos[i].history['val_acc'])

mean = np.mean(val_acc, axis=0)
error = np.std(val_acc, axis=0)

best_epoch = np.argmax(mean) + 1
print('Mean : ', mean)

print('Best acc : ', np.max(mean))
print('Best epoch : ', best_epoch)


# In[ ]:



