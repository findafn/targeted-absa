from Preprocessor import Preprocessor
from Preprocessor import MAX_LENGTH, EMBEDDING_SIZE, ASPECT_LIST

import tensorflow as tf
import numpy as np
import json
import os

from sklearn.metrics import confusion_matrix, f1_score, recall_score, precision_score, accuracy_score
from sklearn.model_selection import train_test_split

import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras_pos_embd import PositionEmbedding
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model, load_model
from keras.layers import Dense, Dropout, Input, Embedding, TimeDistributed
from keras.layers import GRU, LSTM, Bidirectional, GlobalMaxPool1D, Conv1D, MaxPooling1D, GlobalAvgPool1D

class SentimentClassifier():
    config_file = 'config.json'
    weight_file = 'model.h5'
    result_file = 'result.txt'

    def __init__ (
            self,
            module_name = 'sentiment',
            train_file = None,
            test_file = None,
            lowercase = True,
            remove_punct = True,
            embedding = True,
            trainable_embedding = True,
            pos_tag = None,
            dependency = None,
            use_entity = True,
            context = False,
            position_embd = False,
            mask_entity = False,
            use_lexicon = True,
            use_op_target = True,
            use_rnn = True,
            rnn_type = 'lstm',
            return_sequence = True,
            use_cnn = False,
            use_svm = False,
            use_stacked_svm = False,
            use_attention = False,
            n_neuron = 128,
            n_dense = 1,
            dropout = 0.5,
            regularizer = None,
            optimizer = 'adam',
            learning_rate = 0.001,
            weight_decay = 0):
        self.preprocessor  =  Preprocessor(
            module_name = module_name,
            train_file = train_file,
            test_file = test_file,
            lowercase = lowercase,
            remove_punct = remove_punct,
            embedding = embedding,
            pos_tag = pos_tag,
            dependency = dependency,
            use_entity = use_entity,
            context = context,
            position_embd = position_embd,
            mask_entity = mask_entity,
            use_lexicon = use_lexicon,
            use_op_target = use_op_target
        )        
        self.tokenizer = self.preprocessor.get_tokenized()
        self.aspects = ASPECT_LIST
        self.model = None
        self.history = None
        self.result = None
        self.module_name = 'sentiment'

        self.embedding = embedding
        self.trainable_embedding = trainable_embedding
        self.pos_tag = pos_tag
        self.dependency = dependency
        self.use_entity = use_entity
        self.context = context
        self.position_embd = position_embd
        self.mask_entity = mask_entity,
        self.use_lexicon = use_lexicon
        self.use_op_target = use_op_target
        self.use_rnn = use_rnn
        self.rnn_type = rnn_type
        self.use_cnn = use_cnn
        self.use_svm = use_svm
        self.use_stacked_svm = use_stacked_svm
        self.use_attention = use_attention
        self.n_neuron = n_neuron 
        self.n_dense = n_dense 
        self.dropout = dropout
        self.regularizer = regularizer
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.batch_size = None
        self.epochs = None
        self.verbose = None
        self.validation_data = None
        self.cross_validation = None
        self.n_fold = None
        self.grid_search = None
        self.callbacks = None

        self.score = list()
        self.result = {
            'pred' : None,
            'true' : None,
            'join_pred' : None,
            'join_true' : None,
        }

        print("Object has been created")

    def __get_config(self):
        keys = [
            'embedding',
            'trainable_embedding',
            'pos_tag',
            'dependency',
            'use_rnn',
            'rnn_type',
            'use_cnn',
            'use_svm',
            'use_stacked_svm',
            'use_attention',
            'n_neuron',
            'n_dense',
            'dropout',
            'regularizer',
            'optimizer',
            'learning_rate',
            'weight_decay',
            'batch_size',
            'epochs',
            'verbose',
            'cross_validation',
            'n_fold',
            'grid_search',
        ]
        return {k: getattr(self, k) for k in keys}

    def __build_model(self):
        print("Building the model...")
        vocab_size = self.preprocessor.get_vocab_size(self.tokenizer)

        if self.context:
            embedding_matrix = self.preprocessor.get_embedding_matrix(self.tokenizer)
            left_input = Input(shape=(MAX_LENGTH,), dtype='int32', name='left_input')
            x1 = Embedding(
                output_dim=EMBEDDING_SIZE, 
                input_dim=vocab_size, 
                input_length=MAX_LENGTH,
                weights=[embedding_matrix],
                trainable=True,
            )(left_input)

            pos_left_input = Input(shape=(MAX_LENGTH,), dtype='int32', name='pos_left_input')
            x1p = keras.layers.Lambda(
                K.one_hot, 
                arguments={'num_classes': pos_size}, 
                output_shape=(MAX_LENGTH, pos_size)
            )(pos_left_input)

            target_input = Input(shape=(10,), dtype='int32', name='target_input')
            x2 = Embedding(
                output_dim=EMBEDDING_SIZE, 
                input_dim=vocab_size, 
                input_length=10,
                weights=[embedding_matrix],
                trainable=True,
            )(target_input)

            pos_target_input = Input(shape=(10,), dtype='int32', name='pos_target_input')
            x2p = keras.layers.Lambda(
                K.one_hot, 
                arguments={'num_classes': pos_size}, 
                output_shape=(10, pos_size)
            )(pos_target_input)

            right_input = Input(shape=(MAX_LENGTH,), dtype='int32', name='right_input')
            x3 = Embedding(
                output_dim=EMBEDDING_SIZE, 
                input_dim=vocab_size, 
                input_length=MAX_LENGTH,
                weights=[embedding_matrix],
                trainable=True,
            )(right_input)

            pos_right_input = Input(shape=(MAX_LENGTH,), dtype='int32', name='pos_right_input')
            x3p = keras.layers.Lambda(
                K.one_hot, 
                arguments={'num_classes': pos_size}, 
                output_shape=(MAX_LENGTH, pos_size)
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

        else:
            if self.embedding:
                embedding_matrix = self.preprocessor.get_embedding_matrix(self.tokenizer)
                main_input = Input(shape=(MAX_LENGTH,), dtype='int32', name='main_input')
                x = Embedding(
                    output_dim=EMBEDDING_SIZE,
                    input_dim=vocab_size,
                    input_length=MAX_LENGTH,
                    weights=[embedding_matrix],
                    trainable=self.trainable_embedding,
                )(main_input)

                aspect_input = Input(shape=(MAX_LENGTH,), dtype='int32', name='aspect_input')
                x2 = keras.layers.Lambda(
                    K.one_hot, 
                    arguments={'num_classes': len(ASPECT_LIST)}, 
                    output_shape=(MAX_LENGTH, len(ASPECT_LIST))
                )(aspect_input)
                x = keras.layers.concatenate([x, x2])

                if self.position_embd:
                    weights = np.random.random((201, 50))
                    position_input = Input(shape=(MAX_LENGTH,), dtype='int32', name='position_input')
                    x2 = PositionEmbedding(
                        input_shape=(MAX_LENGTH,),
                        input_dim=100,    
                        output_dim=50,     
                        weights=[weights],
                        mode=PositionEmbedding.MODE_EXPAND,
                        name='position_embedding',
                    )(position_input)
                    x = keras.layers.concatenate([x, x2])

                if self.use_lexicon:
                    lex_input = Input(shape=(MAX_LENGTH,), dtype='int32', name='lex_input')
                    x3 = keras.layers.Lambda(
                        K.one_hot, 
                        arguments={'num_classes': 3}, 
                        output_shape=(MAX_LENGTH, 3)
                    )(lex_input)
                    x = keras.layers.concatenate([x, x3])

                if self.pos_tag is 'embedding':
                    _, pos_size = self.preprocessor.get_pos_dict()
                    pos_input = Input(shape=(MAX_LENGTH,), dtype='int32', name='pos_input')
                    x4 = keras.layers.Lambda(
                        K.one_hot, 
                        arguments={'num_classes': pos_size}, 
                        output_shape=(MAX_LENGTH, pos_size)
                    )(pos_input)
                    x = keras.layers.concatenate([x, x4])

            else:
                new_embedding_size = EMBEDDING_SIZE + 6
                if self.pos_tag is 'one_hot':
                    new_embedding_size += 27
                if self.dependency is True:
                    new_embedding_size += 2
                print('embedding size: ', new_embedding_size)
                main_input = Input(shape=(MAX_LENGTH, new_embedding_size), name='main_input')

            print("1. Input")
            
            if self.use_rnn is True:
                if self.embedding is True:
                    if self.rnn_type is 'gru':
                        x = Bidirectional(GRU(self.n_neuron, return_sequences=True))(x)
                    else:
                        x = Bidirectional(LSTM(self.n_neuron, return_sequences=True))(x)
                else:
                    if self.rnn_type is 'gru':
                        x = Bidirectional(GRU(self.n_neuron, return_sequences=True))(main_input)
                    else:
                        x = Bidirectional(LSTM(self.n_neuron, return_sequences=True))(main_input)
                # x = GlobalMaxPool1D()(x)
                x = GlobalAvgPool1D()(x)
                x = Dropout(self.dropout)(x)

            print("2. LSTM")

            if self.use_cnn is True:
                pass

            if self.n_dense is not 0:
                for i in range(self.n_dense):
                    x = Dense(self.n_neuron, activation='relu')(x)
                    x = Dropout(self.dropout)(x)

            print("3. Dense")

            out = Dense(2, activation='softmax')(x)

            print("4. Out")

            x_input = list()
            x_input.append(main_input)
            x_input.append(aspect_input)

            if self.position_embd:
                x_input.append(position_input)
            if self.use_lexicon:
                x_input.append(lex_input)
            if self.pos_tag is 'embedding':
                x_input.append(pos_input)
            
            model = Model(x_input, out)

        print("5. Model")

        model.summary()
        model.compile(
            loss='categorical_crossentropy',
            optimizer=self.optimizer,
            metrics=['acc']
        )

        print("6. Done")

        return model

    def train(
        self,
        x_train,
        y_train,
        batch_size = 16,
        epochs = 5,
        verbose = 1,
        validation_data = False,
        cross_validation = False,
        n_fold = 3,
        grid_search = False,
        callbacks = None):

        self.batch_size = batch_size
        self.epochs = epochs
        self.verbose = verbose
        self.validation_data = validation_data
        self.cross_validation = cross_validation
        self.n_fold = n_fold
        self.grid_search = grid_search
        self.callbacks = callbacks
        
        model = self.__build_model()

        print("Training...")

        x_input = list()
        _, aspect_train = self.preprocessor.read_sentiment(self.preprocessor.train_file, x_train)

        print('x train shape: ', x_train.shape)
        print('aspect train shape: ', aspect_train.shape)

        if self.validation_data:
            input_val = list()
            _, y_train_aspect, _, _ = self.preprocessor.get_all_input_aspect()
            y_train_aspect, y_val_aspect = train_test_split(y_train_aspect, test_size=0.1, random_state=70) 

            idx = 0
            for data in y_train_aspect:
                for asp in data:
                    if asp == 1:
                        idx += 1

            x_val = x_train[idx:]
            y_val = y_train[idx:]
            aspect_val = aspect_train[idx:]

            x_train = x_train[:idx]
            y_train = y_train[:idx]
            aspect_train = aspect_train[:idx]
            
            input_val.append(x_val)
            input_val.append(aspect_val)

        x_input.append(x_train)
        x_input.append(aspect_train)

        print('x train shape new: ', x_train.shape)
        print('aspect train shape new: ', aspect_train.shape)

        print('x val shape new: ', x_val.shape)
        print('aspect val shape new: ', aspect_val.shape)

        if self.position_embd:
            if self.mask_entity:
                position_train = self.preprocessor.get_positional_embedding_with_masking(self.preprocessor.train_file)  
            else:       
                position_train = self.preprocessor.get_positional_embedding_without_masking(self.preprocessor.train_file)

            if self.validation_data:
                position_val = position_train[idx:]
                position_train = position_train[:idx]
                input_val.append(position_val)
            
            x_input.append(position_train)

        if self.use_lexicon:
            posneg_train = self.preprocessor.get_sentiment_lexicons('data/entity_train.json')

            if self.validation_data:
                posneg_val = posneg_train[idx:]
                posneg_train = posneg_train[:idx] 
                input_val.append(posneg_val)
            x_input.append(posneg_train)

        if self.pos_tag == 'embedding':
            pos_train = self.preprocessor.read_pos('resource/postag_train_auto.json')   

            if self.validation_data:
                pos_val = pos_train[idx:]
                pos_train = pos_train[:idx] 
                input_val.append(pos_val)

            x_input.append(pos_train)

        if self.validation_data:
            history = model.fit(
                x = x_input, 
                y = y_train, 
                batch_size = batch_size,
                epochs = epochs, 
                verbose = verbose,
                validation_data = [input_val, y_val],
                callbacks = callbacks
            )
        else:
            history = model.fit(
                x = x_input, 
                y = y_train, 
                batch_size = batch_size,
                epochs = epochs, 
                verbose = verbose,
                callbacks = callbacks
            )

        self.model = model
        self.history = history

        if self.validation_data:
            print("Best epoch : ", np.argmax(self.history.history['val_acc'])+1)

    def change_to_multilabel(self, aspects, sentiments):
        multilabel = list()
        # idx = 0
        # for data in aspects:
        #     temp = list()
        #     for aspect in data:
        #         if aspect == 1:
        #             print(idx)
        #             idx += 1

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

    def read_aspect(self, label):
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

    def change_aspect_to_one_hot(self, label_aspect):
        aspect_train = list()
        label = self.read_aspect(label_aspect)

        for i in range(len(label)):
            for j in range(len(label[i])):
                if label[i][j] == 1:
                    temp = list()
                    length = 110
                    for _ in range(length):
                        temp.append(j)
                    aspect_train.append(temp)
        aspect_train = np.array(aspect_train)
        return aspect_train

    def evaluate(self, x_train, y_train, x_test, y_test):
        _, y_train_aspect, _, y_test_aspect = self.preprocessor.get_all_input_aspect()

        if self.validation_data:
            y_train_aspect, y_val_aspect = train_test_split(y_train_aspect, test_size=0.1, random_state=70)

            idx = 0
            for data in y_train_aspect:
                for asp in data:
                    if asp == 1:
                        idx += 1

            x_val = x_train[idx:]
            y_val = y_train[idx:]

            x_train = x_train[:idx]
            y_train = y_train[:idx]

            x_name = ['Train-All', 'Val-All']
            x = [x_train, x_val, y_train, y_val]
            x_aspect = [y_train_aspect, y_val_aspect]
            y_test_aspect = y_val_aspect
            x_test = x_val
            y_test = y_val
        else:
            x = [x_train, x_test, y_train, y_test]
            x_name = ['Train-All', 'Test-All']
            x_aspect = [y_train_aspect, y_test_aspect]

        print("======================= EVALUATION =======================")
        title = '{:10s} {:10s} {:10s} {:10s} {:10s}'.format('ASPECT', 'ACC', 'PREC', 'RECALL', 'F1')
        self.score.append(title)
        print(title)

        self.evaluate_all(x_train, x_test, y_train, y_test, x_aspect)
        self.evaluate_each_aspect(x_test, y_test, y_test_aspect)

    def evaluate_all(self, x_train, x_test, y_train, y_test, x_aspect):
        x = [x_train, x_test, y_train, y_test]     

        x_name = ['Train-All', 'Test-All'] 

        _, aspect_train = self.preprocessor.read_sentiment(self.preprocessor.train_file, x_train)
        _, y_train_aspect, _, y_test_aspect = self.preprocessor.get_all_input_aspect()

        if self.validation_data:
            y_train_aspect, y_val_aspect = train_test_split(y_train_aspect, test_size=0.1, random_state=70)

            idx = 0
            for data in y_train_aspect:
                for asp in data:
                    if asp == 1:
                        idx += 1

            aspect_val = aspect_train[idx:]
            aspect_train = aspect_train[:idx]

            x_asp = [aspect_train, aspect_val]
        else:
            _, aspect_test = self.preprocessor.read_sentiment(self.preprocessor.test_file, x_test)
            x_asp = [aspect_train, aspect_test]

        if self.position_embd:
            if self.mask_entity:
                position_train = self.preprocessor.get_positional_embedding_with_masking(self.preprocessor.train_file)  
                position_test = self.preprocessor.get_positional_embedding_with_masking(self.preprocessor.test_file)
            else:       
                position_train = self.preprocessor.get_positional_embedding_without_masking(self.preprocessor.train_file)
                position_test = self.preprocessor.get_positional_embedding_without_masking(self.preprocessor.test_file)

            if self.validation_data:
                position_val = position_train[idx:]
                position_train = position_train[:idx]
                x_position = [position_train, position_val] 
            else:
                x_position = [position_train, position_test] 

        if self.use_lexicon:
            posneg_train = self.preprocessor.get_sentiment_lexicons('data/entity_train.json')
            posneg_test = self.preprocessor.get_sentiment_lexicons('data/entity_test.json')

            if self.validation_data:
                posneg_val = posneg_train[idx:]
                posneg_train = posneg_train[:idx]
                x_lex = [posneg_train, posneg_val] 
            else:
                x_lex = [posneg_train, posneg_test] 

        if self.pos_tag is 'embedding':
            pos_train = self.preprocessor.read_pos('resource/postag_train_auto.json')
            pos_test = self.preprocessor.read_pos('resource/postag_test_auto.json')

            if self.validation_data:  
                pos_val = pos_train[idx:]
                pos_train = pos_train[:idx] 

                print('pos train shape: ', pos_train.shape)
                print('pos val shape: ', pos_val.shape)
                x_pos = [pos_train, pos_val] 
            else:
                x_pos = [pos_train, pos_test] 

        for i in range(2):
            x_input = list()
            x_input.append(x[i])
            x_input.append(x_asp[i])

            if self.position_embd:
                x_input.append(x_position[i])
            if self.use_lexicon:
                x_input.append(x_lex[i])
            if self.pos_tag == 'embedding':
                x_input.append(x_pos[i])

            y_pred = self.model.predict(x_input)
            y_true = x[i+2]

            y_pred = np.argmax(y_pred, axis=1)
            y_true = np.argmax(y_true, axis=1)

            y_pred_multilabel = self.change_to_multilabel(x_aspect[i], y_pred)
            y_true_multilabel = self.change_to_multilabel(x_aspect[i], y_true)

            pred_reshape = np.reshape(y_pred_multilabel, -1)
            true_reshape = np.reshape(y_true_multilabel, -1)

            acc = accuracy_score(y_true, y_pred)
            precision = precision_score(true_reshape, pred_reshape, average='macro')
            recall = recall_score(true_reshape, pred_reshape, average='macro')
            f1 = f1_score(true_reshape, pred_reshape, average='macro')

            if i == 1:
                self.result = {
                    'pred' : y_pred,
                    'true' : y_true,
                    'join_pred' : y_pred_multilabel,
                    'join_true' : y_true_multilabel
                }

            score = '{:10s} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f} '.format(x_name[i], acc, precision, recall, f1)
            print(score)
            # print(y_true_multilabel[166], y_true[166])
            # print(y_pred_multilabel[166], y_pred[166])
            self.score.append(score)


    def evaluate_each_aspect(self, x_test, y_test, y_test_aspect):
        y_pred = self.change_to_multilabel(y_test_aspect, self.result['pred'])
        y_true = self.change_to_multilabel(y_test_aspect, self.result['true'])

        true_transpose = y_true.transpose()
        pred_transpose = y_pred.transpose()

        class_names = [0,1,2]
        f1pos = list()
        f1neg = list()
        f1avg = list()

        cnf_matrix = list()
        for i in range(len(self.aspects)):
            cnf_matrix.append(
                confusion_matrix(
                    true_transpose[i], 
                    pred_transpose[i], 
                    labels=class_names)
                )
            np.set_printoptions(precision=2)

        for i in range(len(self.aspects)):
            precpos = cnf_matrix[i][1][1]/(cnf_matrix[i][0][1]+cnf_matrix[i][1][1]+cnf_matrix[i][2][1])
            recpos = cnf_matrix[i][1][1]/(cnf_matrix[i][1][0]+cnf_matrix[i][1][1]+cnf_matrix[i][1][2])
            f1p = 2*(precpos*recpos)/(precpos+recpos)
            f1pos.append(f1p)

            precneg = cnf_matrix[i][2][2]/(cnf_matrix[i][0][2]+cnf_matrix[i][1][2]+cnf_matrix[i][2][2])
            recneg = cnf_matrix[i][2][2]/(cnf_matrix[i][2][0]+cnf_matrix[i][2][1]+cnf_matrix[i][2][2])
            f1n = 2*(precneg*recneg)/(precneg+recneg)
            f1neg.append(f1n) 

            precw = ((precpos*cnf_matrix[i][1][1]) + (precneg*cnf_matrix[i][2][2]))/(cnf_matrix[i][1][1] + cnf_matrix[i][2][2])
            recw = ((recpos*cnf_matrix[i][1][1]) + (recneg*cnf_matrix[i][2][2]))/(cnf_matrix[i][1][1] + cnf_matrix[i][2][2])
            f1w = 2*(precw*recw)/(precw+recw)
            f1avg.append(f1w)
            
            score = '{:10s} {:<10.4f} {:<10.4f} {:<10.4f} {:<10.4f}'.format(self.aspects[i], class_names[0], precw, recw, f1w)
            print(score)
            self.score.append(score)

        print('\n')

    def get_context(self, texts, entity, aspect):
        list_left = list()
        list_right = list()
        list_target = list()

        for i, text in enumerate(texts):   
            print(text)
            left = list()
            right = list()
            target = list()         
            if entity != []:
                split = text.lower().split()             
                e_split = entity[i].split()
                e_first = e_split[0]
                print(split)
                loc = split.index(e_first)

                for j in range(0,loc):
                    left.append(split[j])
                for j in range(len(e_split)):
                    target.append(e_split[j])
                for j in range(loc+len(e_split), len(split)):
                    right.append(split[j])
                
                for _ in aspect[i]:
                    list_left.append(' '.join(left))
                    list_right.append(' '.join(right))
                    list_target.append(' '.join(target))
            else:
                split = text.split()
                left = split
                for _ in aspect[i]:
                    list_left.append(' '.join(left))
                    list_right.append(right)
                    list_target.append(target)

        return list_left, list_target, list_right

    def predict(self, tokenized, aspect, entity):
        print('################# sentiment #################')
        print('input >>>', tokenized)
        print('entity >>>', entity)
        print('aspect >>>', aspect)
        print('length >>>', len(entity)*len(aspect))

        new = list()
        # for i in range(len(tokenized)):
        #     temp = [tokenized[i] for _ in range(len(entity[i])*len(aspect[i]))]
        #     new = new + temp
        if entity != []:
            for i, ent in enumerate(entity):
                new.append(tokenized[0])
        else:
            new = tokenized

        print('input 2 >>>', new)

        left_tr, target_tr, right_tr = self.get_context(new, entity, aspect)
        print(left_tr, '===', target_tr, '===', right_tr)

        new_left = self.tokenizer.texts_to_sequences(left_tr)    
        new_left = pad_sequences(new_left, maxlen=MAX_LENGTH, padding='post')
        new_target = self.tokenizer.texts_to_sequences(target_tr)    
        new_target = pad_sequences(new_target, maxlen=10, padding='post')
        new_right = self.tokenizer.texts_to_sequences(right_tr)    
        new_right = pad_sequences(new_right, maxlen=MAX_LENGTH, padding='post')

        new_aspect = self.change_aspect_to_one_hot(aspect)

        print('left >>', np.array(new_left).shape)
        print('isinya >>',new_left)
        print('target >>', np.array(new_target.shape))
        print('isinya >>',new_target)
        print('right >>', np.array(new_right.shape))
        print('isinya >>',new_right)
        print('aspect >>', np.array(new_aspect.shape))

        y_pred = self.model.predict([np.array(new_left), np.array(new_target), np.array(new_right), np.array(new_aspect)])
        y_pred = np.argmax(y_pred, axis=1)

        print('y_pred >>', y_pred)
        
        label = list()
        if entity != []:
            idx = 0
            for i, ent in enumerate(entity):  
                temp = list()  
                for asp in aspect[i]:
                    if (y_pred[idx] == 1):
                        temp.append('positive')
                    elif (y_pred[idx] == 0):
                        temp.append('negative')
                    idx += 1
                label.append(temp)
        else:
            temp = list()  
            for i, asp in enumerate(aspect[0]):
                if (y_pred[i] == 1):
                    temp.append('positive')
                elif (y_pred[i] == 0):
                    temp.append('negative')
            label.append(temp)

        print("======================= PREDICTION =======================" )
        print(new)
        print(label)
        return label

    def save(self, dir_path):
        if not os.path.exists(dir_path):
            print("Making the directory: {}".format(dir_path))
            os.mkdir(dir_path)
        self.model.save(os.path.join(dir_path, self.weight_file))
        y = self.__get_config()
        with open(os.path.join(dir_path, self.config_file), 'w') as f:
            f.write(json.dumps(y, indent=4, sort_keys=True))
        with open(os.path.join(dir_path, self.result_file), 'w') as f:
            f.write("======================= EVALUATION =======================\n")
            for score in self.score:
                f.write(score + "\n")
            f.write("\n")

        if not self.validation_data:
            review_test = self.preprocessor.read_data_for_aspect('data/entity_test.json')
            entities = self.preprocessor.get_entities('data/entity_test.json')

            with open(os.path.join(dir_path, self.result_file), 'w') as f:
                f.write("======================= EVALUATION =======================\n")
                for score in self.score:
                    f.write(score + "\n")
                f.write("\n")
                f.write("======================= WRONG PREDICTION =======================\n")
                idx = 0
                for i, pred in enumerate(self.result['join_pred']):
                    for j, asp in enumerate(pred):
                        if asp != 0 and asp != self.result['join_true'][i][j]:
                            f.write(str(i) + "\n")
                            f.write(review_test[i] + "\n")
                            if asp == 1:
                                f.write("TRUE: "+ entities[i] + " - " + self.aspects[j] + " - negative\n")
                                f.write("PRED: "+ entities[i] + " - " + self.aspects[j] + " - positive\n")
                            elif asp == 2:
                                f.write("TRUE: "+ entities[i] + " - " + self.aspects[j] + " - positive\n")
                                f.write("PRED: "+ entities[i] + " - " + self.aspects[j] + " - negative\n")
                        
                f.write("======================= TRUE PREDICTION =======================\n")
                i = 0
                for i, pred in enumerate(self.result['join_pred']):
                    for j, asp in enumerate(pred):
                        if asp != 0 and asp == self.result['join_true'][i][j]:
                            f.write(str(i) + "\n")
                            f.write(review_test[i] + "\n")
                            if asp == 1:
                                f.write("TRUE: "+ entities[i] + " - " + self.aspects[j] + " - positive\n")
                                f.write("PRED: "+ entities[i] + " - " + self.aspects[j] + " - positive\n")
                            elif asp == 2:
                                f.write("TRUE: "+ entities[i] + " - " + self.aspects[j] + " - negative\n")
                                f.write("PRED: "+ entities[i] + " - " + self.aspects[j] + " - negative\n")

    def load(self, dir_path, custom):
        if not os.path.exists(dir_path):
            raise OSError('Directory \'{}\' not found.'.format(dir_path))
        else:
            self.model = load_model(os.path.join(dir_path, self.weight_file), custom_objects=custom)
        return self.model