from Preprocessor import Preprocessor
from aspect.AspectCategorizer import AspectCategorizer

import time
import tensorflow as tf
from keras import backend as k
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import ParameterGrid

if __name__ == '__main__':
    config = tf.ConfigProto()
    # Don't pre-allocate memory; allocate as-needed
    config.gpu_options.allow_growth = True
    # Only allow a total of half the GPU memory to be allocated
    config.gpu_options.per_process_gpu_memory_fraction = 0.3

    k.tensorflow_backend.set_session(tf.Session(config=config))

    model = AspectCategorizer(
        train_file = 'data/entity_train.json',
        test_file= 'data/entity_test.json',
        lowercase = True,
        remove_punct = True,
        embedding = True,
        trainable_embedding = False,
        pos_tag = None,
        dependency = False,
        use_entity = True,
        postion_embd = True,
        mask_entity = False,    
        use_rnn = True,
        rnn_type = 'lstm',
        use_cnn = False,
        use_svm = False,
        use_stacked_svm = False,
        use_attention = False,
        n_neuron = 128,
        n_dense = 1,
        dropout = 0.5,
        regularizer = None,
        optimizer = 'adam'
        )

    x_train, y_train, x_test, y_test = model.preprocessor.get_all_input_aspect()

    checkpoint = ModelCheckpoint(
        'aspect/model/callbacks/model_nomask_notrain.h5', 
        monitor='val_f1', 
        verbose=0, 
        save_best_only=True, 
        mode='max', 
        period=1
    )

    model.train(
        x_train, 
        y_train,
        batch_size = 16,
        epochs = 10,
        verbose = 1,
        validation_data = False,
        cross_validation = False,
        n_fold = 10,
        grid_search = False,
        callbacks = checkpoint
        )

    # model.load('aspect/model/nomasking_train')

    # model.evaluate(x_train, y_train, x_test, y_test)
    # model.evaluate_each_aspect(x_test, y_test)

    # named_tuple = time.localtime()
    # time_string = time.strftime("%B-%d-%Y_%H-%M-%S", named_tuple)
    # model.save('aspect/model/{}'.format(time_string))

    # model.predict("Bensin nya irit banget nih tapi sayang kalo buat bepergian jauh mesinnya kurang kuat.")
    # model.predict("Pake mobil ini memang gak pernah kecewa, servis nya cepet sekali")
    print('\n')