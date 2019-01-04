import tensorflow as tf 
import json
import numpy as np 
import os 

class BaseModel():
    def __init__(self,args):
        self.args = args
        self.global_step = tf.Variable(0, dtype = tf.int32, name = "global_step", trainable = False)


    def _create_training_tensors(self):
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = self.logits, labels = self.y))

        optimizer = tf.train.AdamOptimizer(self.args["learning_rate"])
        gradients, v = zip(*optimizer.compute_gradients(self.loss))
        gradients, _ = tf.clip_by_global_norm(gradients, self.args["clip_value"])

        self.train_op = optimizer.apply_gradients(zip(gradients, v),
                                                global_step=self.global_step)
    
    def get_trainable_variables(self):
        """
        Return all trainable variables inside the model
        """
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)

    def show_parameter_count(self, variables):
        """
        Count and print how many parameters there are.
        """
        total_parameters = 0
        for variable in variables:
            name = variable.name

            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parametes = 1
            for dim in shape:
                variable_parametes *= dim.value
            print('{}: {} ({} parameters)'.format(name,
                                                shape,
                                                variable_parametes))
            total_parameters += variable_parametes

        print('Total: {} parameters'.format(total_parameters))
    
    def train(self, session, save_path, train_data, valid_data,
            batch_size, epochs):
        raise NotImplementedError()



    def save(self, saver, session, directory):
        """
        Save the autoencoder model and metadata to the specified
        directory.
        """
        model_path = os.path.join(directory, 'model')
        saver.save(session, model_path)
