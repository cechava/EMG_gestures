import numpy as np
from sklearn.model_selection import train_test_split

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
        self.val_lp_loss = tf.keras.metrics.Mean(name = 'val_lp_loss')
        self.train_lp_accuracy = tf.keras.metrics.CategoricalAccuracy(name = 'lp_acc')
        self.val_lp_accuracy = tf.keras.metrics.CategoricalAccuracy(name = 'val_lp_acc')
        
        self.train_dp_loss = tf.keras.metrics.Mean(name = 'dp_loss')
        self.val_dp_loss = tf.keras.metrics.Mean(name = 'val_dp_loss')
        self.train_dp_accuracy = tf.keras.metrics.CategoricalAccuracy(name = 'dp_acc')
        self.val_dp_accuracy = tf.keras.metrics.CategoricalAccuracy(name = 'val_dp_acc')
        
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
    

    def test_step_label_pred(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self.predict_label(x, training = False)# Forward pass
            lp_loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)   
        #Update metrics
        self.val_lp_loss.update_state(lp_loss)
        self.val_lp_accuracy.update_state(y, y_pred)
         # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    

    def test_step_domain_pred(self, data):
        x, y = data

        with tf.GradientTape() as tape:
            y_pred = self.predict_domain(x, training = False)# Forward pass
            dp_loss = self.compiled_loss(y, y_pred)   
        #Update metrics
        self.val_dp_loss.update_state(dp_loss)
        self.val_dp_accuracy.update_state(y, y_pred)
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
        return [self.train_lp_loss, self.train_lp_accuracy, self.train_dp_loss, self.train_dp_accuracy, \
                self.val_lp_loss, self.val_lp_accuracy ,self.val_dp_loss, self.val_dp_accuracy,]
    
#     def Callback_EarlyStopping(self, LossList, min_delta=0.1, patience=5):
#         #No early stopping for 2*patience epochs 
#         if len(LossList)//patience < 2 :
#             return False
#         #Mean loss for last patience epochs and second-last patience epochs
#         mean_previous = np.mean(LossList[::-1][patience:2*patience]) #second-last
#         mean_recent = np.mean(LossList[::-1][:patience]) #last
#         #you can use relative or absolute change
#         delta_abs = np.abs(mean_recent - mean_previous) #abs change
#         delta_abs = np.abs(delta_abs / mean_previous)  # relative change
#         if delta_abs < min_delta :
#             print("*CB_ES* Loss didn't change much from last %d epochs"%(patience))
#             print("*CB_ES* Percent change in loss value:", delta_abs*1e2)
#             return True
#         else:
#             return False


    def train_label_pred(self, X, Y, validation_split, epochs, batch_size, verbose, callback):
        
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=validation_split, random_state=42)

        # Prepare the training dataset.
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
        val_dataset = val_dataset.shuffle(buffer_size=1024).batch(batch_size)
        
        #callback
        callback.on_train_begin()
        for epoch in range(epochs):
            if verbose:
                print('\n Epoch %d/%d'%(epoch+1,epochs))
            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                out = self.train_step_label_pred((x_batch_train, y_batch_train))

            for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
                out = self.test_step_label_pred((x_batch_val, y_batch_val))
                        
            if verbose:
                log_string = ''
                for k in out.keys():
                    log_string = log_string + ' - %s: %.04f '%(k,out[k])
                print(log_string)
            
            #callback
            callback.on_epoch_end(epoch, out['val_lp_loss'])
            #reset states after every epoch
            self.train_lp_loss.reset_states()
            self.val_lp_loss.reset_states()
            self.train_lp_accuracy.reset_states()
            #check for early stopping flag
            if callback.stop_training:
                print('Early Stopping after %d epochs'%(epoch))
                break
        callback.on_train_end()
        
    def train_domain_adapt(self, X, Y, X_domain, validation_split, epochs, batch_size, verbose, callback):
            
        X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=validation_split, random_state=42)
        # Prepare the training dataset.
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
        val_dataset = val_dataset.shuffle(buffer_size=1024).batch(batch_size)
        
        X_combo = np.vstack((X,X_domain))
        Y_combo_domain = np.hstack((np.ones(X.shape[0],)*0,np.ones(X_domain.shape[0],)*1))#source_label
        Y_combo_domain = to_categorical(Y_combo_domain)
        
        X_domain_train, X_domain_val, Y_domain_train, Y_domain_val = train_test_split(X_combo, Y_combo_domain, test_size=validation_split, random_state=42)


        train_domain_dataset = tf.data.Dataset.from_tensor_slices((X_domain_train, Y_domain_train))
        train_domain_dataset = train_domain_dataset.shuffle(buffer_size=1024).batch(batch_size)
        
        val_domain_dataset = tf.data.Dataset.from_tensor_slices((X_domain_val, Y_domain_val))
        val_domain_dataset = val_domain_dataset.shuffle(buffer_size=1024).batch(batch_size)
        
        #callback
        callback.on_train_begin()
        for epoch in range(epochs):
            if verbose:
                print('\n Epoch %d/%d'%(epoch+1,epochs))
            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                out = self.train_step_label_pred((x_batch_train, y_batch_train))
                
            for step, (x_domain_batch_train,y_domain_batch_train) in enumerate(train_domain_dataset):
                out = self.train_step_domain_adapt((x_domain_batch_train, y_domain_batch_train))
                
            for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
                out = self.test_step_label_pred((x_batch_val, y_batch_val))
                
            for step, (x_domain_batch_val,y_domain_batch_val) in enumerate(val_domain_dataset):
                out = self.test_step_domain_pred((x_domain_batch_val, y_domain_batch_val))
                
                
            #output metrics
            if verbose:
                log_string = ''
                for k in out.keys():
                    log_string = log_string + ' - %s: %.04f '%(k,out[k])
                print(log_string)
                
            #callback
            #print(out['val_dp_acc'], self.val_dp_accuracy.result())
            callback.on_epoch_end(epoch, out['val_dp_loss'])
            
            #reset states after every epoch
            self.train_lp_loss.reset_states()
            self.val_lp_loss.reset_states()
            self.train_lp_accuracy.reset_states()
            self.val_lp_accuracy.reset_states()
            self.train_dp_loss.reset_states()
            self.val_dp_loss.reset_states()
            self.train_dp_accuracy.reset_states()
            self.val_dp_accuracy.reset_states()
            
            #check for early stopping flag
            if callback.stop_training:
                print('Early Stopping after %d epochs'%(epoch))
                break
        callback.on_train_end()
        
        return
    def train_domain_and_labels(self, X, Y, X_domain, Y_domain, validation_split, epochs, batch_size, verbose, callback):
        
        X_all = np.vstack((X,X_domain))
        Y_all = np.vstack((Y,Y_domain))
        Y_combo_domain = np.hstack((np.ones(X.shape[0],)*0,np.ones(X_domain.shape[0],)*1))#source_label
        Y_combo_domain = to_categorical(Y_combo_domain)
        
        X_train, X_val, Y_train, Y_val = train_test_split(X_all, Y_all, test_size=validation_split, random_state=42)
        # Prepare the training dataset.
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
        train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)
        
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, Y_val))
        val_dataset = val_dataset.shuffle(buffer_size=1024).batch(batch_size)
        
        X_domain_train, X_domain_val, Y_domain_train, Y_domain_val = train_test_split(X_all, Y_combo_domain, test_size=validation_split, random_state=42)


        train_domain_dataset = tf.data.Dataset.from_tensor_slices((X_domain_train, Y_domain_train))
        train_domain_dataset = train_domain_dataset.shuffle(buffer_size=1024).batch(batch_size)
        
        val_domain_dataset = tf.data.Dataset.from_tensor_slices((X_domain_val, Y_domain_val))
        val_domain_dataset = val_domain_dataset.shuffle(buffer_size=1024).batch(batch_size)

        #callback
        callback.on_train_begin()
        for epoch in range(epochs):
            if verbose:
                print('\n Epoch %d/%d'%(epoch+1,epochs))
            # Iterate over the batches of the dataset.
            for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
                out = self.train_step_label_pred((x_batch_train, y_batch_train))
                
            for step, (x_domain_batch_train,y_domain_batch_train) in enumerate(train_domain_dataset):
                out = self.train_step_domain_adapt((x_domain_batch_train, y_domain_batch_train))
                
            for step, (x_batch_val, y_batch_val) in enumerate(val_dataset):
                out = self.test_step_label_pred((x_batch_val, y_batch_val))
                
            for step, (x_domain_batch_val,y_domain_batch_val) in enumerate(val_domain_dataset):
                out = self.test_step_domain_pred((x_domain_batch_val, y_domain_batch_val))

            #output metrics
            if verbose:
                log_string = ''
                for k in out.keys():
                    log_string = log_string + ' - %s: %.04f '%(k,out[k])
                print(log_string)
                
            #callback
            callback.on_epoch_end(epoch, out['val_lp_loss'])
            
            #reset states after every epoch
            self.train_lp_loss.reset_states()
            self.val_lp_loss.reset_states()
            self.train_lp_accuracy.reset_states()
            self.val_lp_accuracy.reset_states()
            self.train_dp_loss.reset_states()
            self.val_dp_loss.reset_states()
            self.train_dp_accuracy.reset_states()
            self.val_dp_accuracy.reset_states()
            
            #check for early stopping flag
            if callback.stop_training:
                print('Early Stopping after %d epochs'%(epoch))
                break
        callback.on_train_end()
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

class EarlyStopping_Custom():
    def __init__(self,
               min_delta=0,
               patience=0,
               verbose=0,
               mode='auto',
               baseline=None):
        super(EarlyStopping_Custom, self).__init__()
        self.patience = patience
        self.verbose = verbose
        self.baseline = baseline
        self.min_delta = abs(min_delta)
        self.wait = 0
        self.stopped_epoch = 0


        if mode not in ['auto', 'min', 'max']:
            logging.warning('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.', mode)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less

        if self.monitor_op == np.greater:
            self.min_delta *= 1
        else:
            self.min_delta *= -1

    def on_train_begin(self):
        # Allow instances to be re-used
        self.wait = 0
        self.stopped_epoch = 0
        self.stop_training = False
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf
        self.best_weights = None

    def on_epoch_end(self, epoch, current):
        if current is None:
            return
        self.wait += 1
        if self._is_improvement(current, self.best):
            self.best = current
            # Only restart wait if we beat our previous best.
            if self.baseline is None or self._is_improvement(current, self.baseline):
                self.wait = 0

        if self.wait >= self.patience:
            self.stopped_epoch = epoch
            self.stop_training = True

    def on_train_end(self):
        if self.stopped_epoch > 0 and self.verbose > 0:
              print('Epoch %05d: early stopping' % (self.stopped_epoch + 1))

    def _is_improvement(self, monitor_value, reference_value):
        return self.monitor_op(monitor_value-self.min_delta , reference_value)
