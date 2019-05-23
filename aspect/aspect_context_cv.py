
# coding: utf-8

# In[14]:

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
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import classification_report

import keras
from keras import backend as k
from keras.models import Model, load_model
from keras.layers import GRU, LSTM, Bidirectional, GlobalMaxPool1D, Conv1D, MaxPooling1D, GlobalAveragePooling1D
from keras.layers import Dense, Dropout, Input, Embedding, TimeDistributed
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences


# In[2]:

with open('../aspect_detection/w2v_path.txt') as file:
    word2vec_path = file.readlines()[0]
    
w2v = Word2Vec.load(word2vec_path)


# In[23]:

config = tf.ConfigProto()
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.3

k.tensorflow_backend.set_session(tf.Session(config=config))

class Metrics(keras.callbacks.Callback):
    def on_epoch_end(self, batch, logs={}):
        predict = np.asarray(self.model.predict(self.validation_data[0]))
        targ = self.validation_data[1]
        self.precisions= precision_score(targ, (predict>0.5).astype(int), average='micro')
        self.recalls= recall_score(targ, (predict>0.5).astype(int), average='micro')
        self.f1s= f1_score(targ, (predict>0.5).astype(int), average='micro')

        print("Precision Score :", self.precisions)
        print("Recall Score :", self.recalls)
        print("F1 Score :", self.f1s)
        return
    
metrics = Metrics()

def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
    possible_positives = k.sum(k.round(k.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + k.epsilon())
    return recall

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
    predicted_positives = k.sum(k.round(k.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + k.epsilon())
    return precision

def f1(y_true, y_pred):
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
        possible_positives = k.sum(k.round(k.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + k.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = k.sum(k.round(k.clip(y_true * y_pred, 0, 1)))
        predicted_positives = k.sum(k.round(k.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + k.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)

    return 2*((precision*recall)/(precision+recall+k.epsilon()))


# In[5]:

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
    
    
def read_data_for_aspect(entity_file):
    review = list()
    for data in entity_file:
        for _ in data['info']:
            clean = re.sub(r"[,.;@?!&$]+\ *", " ", data['masked_sentence'])
            review.append(clean.lower())
    return review

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

    return np.array(encoded_label)

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
                entity = ent['entity_name']
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
            
                list_left.append(' '.join(left))
                list_right.append(' '.join(right))
                list_target.append(' '.join(target))
            else:
                split = review_a[idx].split()
                idx += 1    
                left = split
                list_left.append(' '.join(left))
                list_right.append(right)
                list_target.append(target)

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

left_tr, target_tr, right_tr = get_context(review_atr, entity_train)
left_te, target_te, right_te = get_context(review_ate, entity_test)

y_train = read_aspect(entity_train)
y_test = read_aspect(entity_test)

with open('../../../targeted-absa/tokenizer_2.pickle', 'rb') as handle:
    token = pickle.load(handle)
    
vocab_size = len(token.word_index) + 1

train_left = get_tokenized(left_tr, max_length)
train_target = get_tokenized(target_tr, 10)
train_right = get_tokenized(right_tr, max_length)

test_left = get_tokenized(left_te, max_length)
test_target = get_tokenized(target_te, 10)
test_right = get_tokenized(right_te, max_length)

embedding_matrix = get_embedding(w2v, vocab_size)


# In[31]:

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
#     xp = keras.layers.concatenate([x1p, x2p, x3p], axis=1)
#     x = keras.layers.concatenate([x, xp])

    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    out = Dense(len(ASPECT_LIST), activation='sigmoid')(x)

#     model = Model([left_input, target_input, right_input, pos_left_input, pos_target_input, pos_right_input], out)
    model = Model([left_input, target_input, right_input], out)

    model.summary()

    model.compile(
        loss='binary_crossentropy', 
        optimizer='adam', 
        metrics=[f1]
    )
    return model


# In[54]:

# param_grid = {
#     'batch_size': [32], 
#     'epochs' : [5,6]
# }

# grid = ParameterGrid(param_grid)

# for params in grid:
#     print(params)
#     print('model_batch-{}_epochs-{}.h5'.format(params['batch_size'], params['epochs']))


# In[52]:

# print('model_batch-{}_epochs-{}'.format(params['batch_size'], params['epochs']))


# In[52]:

# train_left, val_left = train_test_split(train_left, test_size=0.1, random_state=70)
# train_target, val_target = train_test_split(train_target, test_size=0.1, random_state=70)
# train_right, val_right = train_test_split(train_right, test_size=0.1, random_state=70)

# pos_train_left, pos_val_left = train_test_split(pos_train_left, test_size=0.1, random_state=70)
# pos_train_target, pos_val_target = train_test_split(pos_train_target, test_size=0.1, random_state=70)
# pos_train_right, pos_val_right = train_test_split(pos_train_right, test_size=0.1, random_state=70)

# y_train, y_val = train_test_split(y_train, test_size=0.1, random_state=70)


# In[11]:

checkpoint = ModelCheckpoint(
    'model/crossval/model_aspect_context.h5', 
    monitor='val_f1', 
    verbose=0, 
    save_best_only=True, 
    mode='max', 
    period=1
)


# In[17]:

# x_input = [train_left, train_target, train_right, pos_train_left, pos_train_target, pos_train_right]
x_input = [train_left, train_target, train_right]


# In[32]:

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

val_f1 = np.zeros((len(histos), len(histos[0].history['val_f1'])))
val_precision = np.zeros((len(histos), len(histos[0].history['val_precision'])))
val_recall = np.zeros((len(histos), len(histos[0].history['val_recall'])))
for i in range(len(histos)):
    print(histos[i].history['val_f1'])
    val_f1[i]= np.asarray(histos[i].history['val_f1'])
    val_precision[i]= np.asarray(histos[i].history['val_precision'])
    val_recall[i]= np.asarray(histos[i].history['val_recall'])

mean = np.mean(val_f1, axis=0)
mean_p = np.mean(val_precision, axis=0)
mean_r = np.mean(val_recall, axis=0)
error = np.std(val_f1, axis=0)

best_epoch = np.argmax(mean) + 1
print('Mean : ', mean)

print('Best precision : ', np.max(mean_p))
print('Best recall : ', np.max(mean_r))
print('Best f1 : ', np.max(mean))
print('Best epoch : ', best_epoch)


# In[108]:

# history = model.fit(
#     x=[train_left, train_target, train_right, pos_train_left, pos_train_target, pos_train_right], 
#     y=y_train, 
#     batch_size=32,
#     epochs=1, 
#     verbose=1,
# #     validation_data=[[val_left, val_target, val_right, pos_val_left, pos_val_target, pos_val_right], y_val]
# )


# In[107]:

# model.save('aspect_context_pos_94.h5')


# In[77]:

# aspect_model = load_model('aspect_context_pos_91.h5', custom_objects={'f1':f1, 'tf':tf})


# In[109]:

# # print("best epoch : ", np.argmax(history.history['val_f1'])+1)

# x1 = [train_left, test_left]
# x2 = [train_target, test_target]
# x3 = [train_right, test_right]

# x1p = [train_left, test_left]
# x2p = [train_target, test_target]
# x3p = [train_right, test_right]

# y = [y_train, y_test]

# print("LSTM Evaluation")

# for i in range(2):
#     y_pred = model.predict([x1[i], x2[i], x3[i], x1p[i], x2p[i], x3p[i]])
#     y_true = y[i]

#     y_pred = np.asarray(y_pred)
#     y_true = np.asarray(y_true)

#     y_pred = (y_pred>0.5).astype(int)
#     y_true = (y_true>0).astype(int)

#     acc = accuracy_score(y_true.reshape([-1]), y_pred.reshape([-1]))
#     precision = precision_score(y_true, y_pred, average='macro')
#     recall = recall_score(y_true, y_pred, average='macro')
#     f1 = f1_score(y_true, y_pred, average='macro')
    
#     print('Evaluasi', i)
#     print('Akurasi : ', acc)
#     print('Precision : ', precision)
#     print('Recall : ', recall)
#     print('F1 : ', f1)
    


# In[60]:

# # print("best epoch : ", np.argmax(history.history['val_f1'])+1)

# x1 = [train_left, val_left]
# x2 = [train_target, val_target]
# x3 = [train_right, val_right]

# x1p = [pos_train_left, pos_val_left]
# x2p = [pos_train_target, pos_val_target]
# x3p = [pos_train_right, pos_val_right]

# y = [y_train, y_val]

# print("LSTM Evaluation")

# for i in range(2):
#     y_pred = model.predict([x1[i], x2[i], x3[i], x1p[i], x2p[i], x3p[i]])
#     y_true = y[i]

#     y_pred = np.asarray(y_pred)
#     y_true = np.asarray(y_true)

#     y_pred = (y_pred>0.5).astype(int)
#     y_true = (y_true>0).astype(int)

#     acc = accuracy_score(y_true.reshape([-1]), y_pred.reshape([-1]))
#     precision = precision_score(y_true, y_pred, average='macro')
#     recall = recall_score(y_true, y_pred, average='macro')
#     f1 = f1_score(y_true, y_pred, average='macro')
    
#     print('Evaluasi', i)
#     print('Akurasi : ', acc)
#     print('Precision : ', precision)
#     print('Recall : ', recall)
#     print('F1 : ', f1)


# In[ ]:




# In[ ]:



