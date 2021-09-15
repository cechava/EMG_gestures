import numpy as np

from tensorflow import keras
from tensorflow.keras.models import Sequential, Model, Sequential
from tensorflow. keras.layers import Dense, Activation, Dropout, Input,  TimeDistributed, GRU, Masking, LSTM
from tensorflow.keras.utils import to_categorical
import tensorflow as tf

class DANN(keras.Model):
    def __init__(self, input_shape, n_outputs, n_source = 2, fe_layers = 0, dp_layers = 0,\
                           drop_prob = 0.5, activation = 'tanh'):
        super(DANN, self).__init__()
        
        #define feature extractor model
        fe_input = Input(shape = input_shape)
        X = fe_input
        for n in range(fe_layers):
            X = Dense(input_shape[0],activation = activation)(X)
            X = Dropout(drop_prob)(X)
        fe_output = X   
        self.feat_extract = Model(inputs = fe_input, outputs = fe_output, name = 'feature_extractor')

        #label-predicting head
        lp_head_input = Input(shape = input_shape)
        lp_head_output = Dense(n_outputs,activation = 'softmax', name = 'label')(lp_head_input)
        self.label_pred = Model(inputs = lp_head_input, outputs = lp_head_output, name = 'label_head')
        
        #domain predicting head
        dp_head_input = Input(shape = input_shape)
        X = GradientReversalLayer()(dp_head_input)#reverse gradient
        for n in range(dp_layers):
            X = Dense(input_shape[0],activation = activation)(X)
            X = Dropout(drop_prob)(X)
        dp_head_output = Dense(n_source,activation = 'softmax', name = 'domain')(X)
        self.domain_pred = Model(inputs = dp_head_input, outputs = dp_head_output, name = 'domain_head')
        

        lp_input = Input(shape = input_shape)
        X = self.feat_extract(lp_input)
        lp_output = self.label_pred(X)
        self.predict_label = Model(inputs = lp_input, outputs = lp_output, name = 'DANN-label')
        
        dp_input = Input(shape = input_shape)
        X = self.feat_extract(dp_input)
        dp_output = self.domain_pred(X)
        self.predict_domain = Model(inputs = dp_input, outputs = dp_output, name = 'DANN-domain')
        
        self.optimizer = tf.keras.optimizers.Adam()
        
        self.train_lp_loss = tf.keras.metrics.Mean(name = 'lp_loss')
        self.train_lp_accuracy = tf.keras.metrics.CategoricalAccuracy(name = 'lp_acc')
        
        self.train_dp_loss = tf.keras.metrics.Mean(name = 'dp_loss')
        self.train_dp_accuracy = tf.keras.metrics.CategoricalAccuracy(name = 'dp_acc')
        
    @tf.function
    def train_step_label_pred(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self.predict_label(x, training = True)# Forward pass
            lp_loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)   
        lp_grad = tape.gradient(lp_loss, self.predict_label.trainable_variables)

        self.optimizer.apply_gradients(zip(lp_grad, self.predict_label.trainable_variables))
        
        # Update metrics
        self.train_lp_loss.update_state(lp_loss)
        self.train_lp_accuracy.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    
    def train_step_domain_adapt(self, data):
        x, y = data


        with tf.GradientTape() as tape:
            y_pred = self.predict_domain(x, training = True)# Forward pass
            dp_loss = self.compiled_loss(y, y_pred)
        dp_grad = tape.gradient(dp_loss, self.predict_domain.trainable_variables)
        
        self.optimizer.apply_gradients(zip(dp_grad, self.predict_domain.trainable_variables))
        
        # Update metrics
        self.train_dp_loss.update_state(dp_loss)
        self.train_dp_accuracy.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        # or at the start of `evaluate()`.
        return [self.train_lp_loss, self.train_lp_accuracy, self.train_dp_loss, self.train_dp_accuracy]

    def train_label_pred(self, X,Y, epochs, batch_size, verbose):

        # Prepare the training dataset.
        train_dataset = tf.data.Dataset.from_tensor_slices((X, Y))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
        for epoch in range(epochs):
            if verbose:
                print('\n Epoch %d/%d'%(epoch+1,epochs))
            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                out = self.train_step_label_pred((x_batch_train, y_batch_train))

            #output metrics
            if verbose:
                log_string = ''
                for k in out.keys():
                    log_string = log_string + ' - %s: %.04f '%(k,out[k])
                print(log_string)
            #reset states after every epoch
            self.train_lp_loss.reset_states()
            self.train_lp_accuracy.reset_states()
            
    def train_domain_adapt(self, X, Y, X_domain, epochs, batch_size, verbose):
        
        X_combo = np.vstack((X,X_domain))
        Y_combo_domain = np.hstack((np.ones(X.shape[0],)*0,np.ones(X_domain.shape[0],)*1))#source_label
        Y_combo_domain = to_categorical(Y_combo_domain)

        # Prepare the training dataset.
        train_dataset = tf.data.Dataset.from_tensor_slices((X, Y))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
        
        train_domain_dataset = tf.data.Dataset.from_tensor_slices((X_combo, Y_combo_domain))
        train_domain_dataset = train_domain_dataset.shuffle(buffer_size=1024).batch(batch_size)
        
        
        for epoch in range(epochs):
            if verbose:
                print('\n Epoch %d/%d'%(epoch+1,epochs))
            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                out = self.train_step_label_pred((x_batch_train, y_batch_train))
                
#                 for x_domain_batch,y_domain_batch in train_domain_dataset.take(1):
#                     out = self.train_step_domain_adapt((x_domain_batch, y_domain_batch))

                
            for step, (x_domain_batch,y_domain_batch) in enumerate(train_domain_dataset):
                out = self.train_step_domain_adapt((x_domain_batch, y_domain_batch))

            #output metrics
            if verbose:
                log_string = ''
                for k in out.keys():
                    log_string = log_string + ' - %s: %.04f '%(k,out[k])
                print(log_string)
                
            #reset states after every epoch
            self.train_lp_loss.reset_states()
            self.train_lp_accuracy.reset_states()
            self.train_dp_loss.reset_states()
            self.train_dp_accuracy.reset_states()
        
        return
    def train_domain_and_labels(self, X, Y, X_domain, Y_domain, epochs, batch_size, verbose):
        
        X_all = np.vstack((X,X_domain))
        Y_all = np.vstack((Y,Y_domain))
        Y_combo_domain = np.hstack((np.ones(X.shape[0],)*0,np.ones(X_domain.shape[0],)*1))#source_label
        Y_combo_domain = to_categorical(Y_combo_domain)

        # Prepare the training dataset.
        train_dataset = tf.data.Dataset.from_tensor_slices((X_all, Y_all))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
        
        train_domain_dataset = tf.data.Dataset.from_tensor_slices((X_all, Y_combo_domain))
        train_domain_dataset = train_domain_dataset.shuffle(buffer_size=1024).batch(batch_size)
        
        
        for epoch in range(epochs):
            if verbose:
                print('\n Epoch %d/%d'%(epoch+1,epochs))
            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                out = self.train_step_label_pred((x_batch_train, y_batch_train))
                

            for step, (x_domain_batch,y_domain_batch) in enumerate(train_domain_dataset):
                out = self.train_step_domain_adapt((x_domain_batch, y_domain_batch))

            #output metrics
            if verbose:
                log_string = ''
                for k in out.keys():
                    log_string = log_string + ' - %s: %.04f '%(k,out[k])
                print(log_string)
                
            #reset states after every epoch
            self.train_lp_loss.reset_states()
            self.train_lp_accuracy.reset_states()
            self.train_dp_loss.reset_states()
            self.train_dp_accuracy.reset_states()
        
        return

class shallow_NN(keras.Model):
    def __init__(self, input_shape, n_outputs, n_dense_pre = 0,\
                           drop_prob = 0.5, activation = 'tanh'):
        super(shallow_NN, self).__init__()
        #define feature extractor model
        fe_input = Input(shape = input_shape)
        X = fe_input
        for n in range(n_dense_pre):
            X = Dense(input_shape[0],activation = activation)(X)
            X = Dropout(drop_prob)(X)
        fe_output = X   
        self.feat_extract = Model(inputs = fe_input, outputs = fe_output, name = 'feature_extractor')

        #label-predicting head
        lp_head_input = Input(shape = input_shape)
        lp_head_output = Dense(n_outputs,activation = 'softmax', name = 'label')(lp_head_input)
        self.label_pred = Model(inputs = lp_head_input, outputs = lp_head_output, name = 'label_predictor')

        lp_input = Input(shape = input_shape)
        X = self.feat_extract(lp_input)
        lp_output = self.label_pred(X)
        self.predict_label = Model(inputs = lp_input, outputs = lp_output, name = 'NN')
        
        self.optimizer = tf.keras.optimizers.Adam()
        
        self.train_loss = tf.keras.metrics.Mean()
        self.train_accuracy = tf.keras.metrics.CategoricalAccuracy()
        
    @tf.function
    def train_step(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self.predict_label(x, training = True)# Forward pass
            lp_loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses,)   
        lp_grad = tape.gradient(lp_loss, self.predict_label.trainable_variables)

        self.optimizer.apply_gradients(zip(lp_grad, self.predict_label.trainable_variables))
        
        # Update metrics
        self.train_loss.update_state(lp_loss)
        self.train_accuracy.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


    @property
    def metrics(self):
        # We list our `Metric` objects here so that `reset_states()` can be
        # called automatically at the start of each epoch
        return [self.train_loss, self.train_accuracy]

#define custom gradient
@tf.custom_gradient
def GradientReversalOperator(x):
    def grad(dy):
        return -1*dy
    return x, grad

class GradientReversalLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(GradientReversalLayer, self).__init__()

    def call(self, inputs):
    	return GradientReversalOperator(inputs)
