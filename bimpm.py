import tensorflow as tf 
import numpy as np 
import os
import argparse
import json
from basemodel import BaseModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils import data_generator, shuffle_data
class BIMPM(BaseModel):
    def __init__(self, args):
        super(BIMPM, self).__init__(args)
        self.args = args
        self.d = args["word_dim"] + int(args["use_char_emb"]) * args["char_hidden_size"]
        self.l = args["num_perspective"]
        self.is_train = args.get("train", False)
        
        # ----- Add Placeholders -----
        self.p = tf.placeholder(dtype = tf.int32, shape = [None, None], name = "premise")
        self.h = tf.placeholder(dtype = tf.int32, shape = [None, None], name = "hypothesis")
        self.char_p = tf.placeholder(dtype = tf.int32, shape = [None, None, args["max_word_len"]], name = "char_premise")
        self.char_h = tf.placeholder(dtype = tf.int32, shape = [None, None, args["max_word_len"]], name = "char_hypothesis")
        self.y = tf.placeholder(dtype = tf.int32, shape = [None], ) 
        self.dropout_rate = tf.placeholder(dtype = tf.float32, shape = [], name = "dropout_rate")
        
        
        # ----- Word Representation Layer -----
        char_emb = tf.get_variable("char_emb", 
            shape = [args["char_vocab_size"], args["char_dim"]], 
            dtype = tf.float32, 
            initializer = tf.random_uniform_initializer(-0.005, 0.005)
        )
        pad = tf.constant(np.zeros([1,args["char_dim"]]), dtype = tf.float32)
        self.char_emb = tf.concat([pad, char_emb], axis = 0)

        self.word_emb = tf.get_variable("word_emb", 
            shape = [args["word_vocab_size"], args["word_dim"]], 
            dtype = tf.float32,
            initializer = tf.random_uniform_initializer(-0.005, 0.005)
        )


        self.char_LSTM = tf.nn.rnn_cell.LSTMCell(
            args["char_hidden_size"],
            initializer = tf.initializers.he_normal(), 
            
        )


        # ----- Context Representation Layer -----
        self.context_LSTM_fw = tf.nn.rnn_cell.LSTMCell(
            args["hidden_size"],
            initializer = tf.initializers.he_normal(),
            
        )
        self.context_LSTM_bw = tf.nn.rnn_cell.LSTMCell(
            args["hidden_size"],
            initializer = tf.initializers.he_normal(),
            
        )


        # ----- Matching Layer -----
        # with tf.variable_scope("weight", reuse = tf.AUTO_REUSE):
        for i in range(1,9):
            setattr(self, 
                f'mp_w{i}', 
                tf.get_variable("mp_w{}".format(i), 
                    shape = [self.l, args["hidden_size"]], 
                    initializer = tf.initializers.he_normal()
                )
            )

        # ----- Aggregation Layer -----
        self.aggregation_LSTM_fw = tf.nn.rnn_cell.LSTMCell(
            args["hidden_size"],
            initializer = tf.initializers.he_normal(),
            
        )
        self.aggregation_LSTM_bw = tf.nn.rnn_cell.LSTMCell(
            args["hidden_size"],
            initializer = tf.initializers.he_normal(),
            
        )

        # ----- Prediction Layer -----
        self.pred_fc1 = tf.layers.Dense(args["hidden_size"] * 2, 
            activation = tf.tanh, 
            kernel_initializer = tf.random_uniform_initializer(-0.005, 0.005),
            bias_initializer = tf.constant_initializer(value = 0.0, dtype = tf.float32)
        )
        self.pred_fc2 = tf.layers.Dense(args["class_size"],
            kernel_initializer = tf.random_uniform_initializer(-0.005, 0.005),
            bias_initializer = tf.constant_initializer(value = 0.0, dtype = tf.float32)
        )


        def dropout(v):
            return tf.nn.dropout(v, keep_prob = 1. - self.dropout_rate)
        
        def cosine_similarity(lfs, rhs):
            """
            :params lfs: [...,d]
            :params hfs: [...,d]
            :return [...]
            """
            dot = tf.reduce_sum(lfs * rhs, axis=-1)
            base = tf.sqrt(tf.reduce_sum(tf.square(lfs), axis=-1)) * tf.sqrt(tf.reduce_sum(tf.square(rhs), axis=-1))
            # return dot / base
            return div_with_small_value(dot, base, eps = 1e-8)

        def div_with_small_value(dot, base, eps = 1e-8):
            # too small values are replaced by 1e-8 to prevent it from exploding.
            eps = tf.ones_like(base) * eps
            base = tf.where(base > eps, base, eps)
            return dot / base

        def mp_matching_func(v1, v2, w):
            """
            :params v1: (batch, seq_len, hidden_size)
            :params v2: (batch, seq_len, hidden_size) or (batch, hidden_size)
            :param w: (l, hidden_size)
            :return: (batch, l)
            """
            v1 = tf.expand_dims(v1, axis = 2) * w # (batch, seq_len, l, hidden_size)

            if v2.shape.ndims == 3: # (batch, seq_len, hidden_size)
                v2 = tf.expand_dims(v2, axis = 2) * w # (batch, seq_len, 1, hidden_size)
            else:
                v2 = tf.expand_dims(tf.expand_dims(v2, axis = 1), axis = 1) * w # (batch, 1, 1, hidden_size)

            m = cosine_similarity(v1, v2)
            return m

        def mp_matching_func_pairwise(v1, v2, w):
            """
            :params v1: (batch, seq_len1, hidden_size)
            :params v2: (batch, seq_len2, hidden_size)
            :params w: (l, hidden_size)
            :return: (batch, l, seq_len1, seq_len2)
            """
            w = tf.expand_dims(w, axis = 1) # (l, 1, hidden_size)
            v1 = tf.expand_dims(v1, axis = 1)  # (batch, 1, seq_len, hidden_size)
            v2 = tf.expand_dims(v2, axis = 1)
            v1 = v1 * w # (batch, l, seq_len, hidden_size)
            v2 = v2 * w
            
            v1_norm = tf.norm(v1, axis = 3, keepdims = True) # (batch, l, seq_len, 1)
            v2_norm = tf.norm(v2, axis = 3, keepdims = True)
            
            n = tf.matmul(v1, v2, transpose_b = True) # (batch, l, seq_len1, seq_len2)
            d = v1_norm * tf.transpose(v2_norm , [0,1,3,2]) # (batch, l, seq_len1, 1) * (batch, l, 1, seq_len2) 
            
            m = div_with_small_value(n, d) # (batch, l, seq_len1, seq_len2)
            m = tf.transpose(m, [0,2,3,1])

            return m

        def attention(v1, v2):
            """
            :param v1: (batch, seq_len1, hidden_size)
            :param v2: (batch, seq_len2, hidden_size)
            :return: (batch, seq_len1, seq_len2)
            """
            v1_norm = tf.norm(v1, axis = 2, keepdims = True)
            v2_norm = tf.norm(v2, axis = 2, keepdims = True)

            # (batch, seq_len1, seq_len2)
            a = tf.matmul(v1_norm, v2_norm, transpose_b = True)
            d = v1_norm * tf.transpose(v2_norm, [0,2,1])
            
            return div_with_small_value(a, d)


        # ----- Word Representation Layer -----
        # (batch, seq_len) -> (batch, seq_len, word_dim)
        p = tf.nn.embedding_lookup(self.word_emb, self.p)
        h = tf.nn.embedding_lookup(self.word_emb, self.h)

        if self.args["use_char_emb"]:
            # (batch, seq_len, word_len) -> (batch * seq_len, max_word_len)
            seq_len_p = tf.shape(self.char_p)[1]
            seq_len_h = tf.shape(self.char_h)[1]

            char_p = tf.reshape(self.char_p, [-1, args.max_word_len])
            char_h = tf.reshape(self.char_h, [-1, args.max_word_len])

            # (batch * seq_len, max_word_len, char_dim) -> (batch * seq_len, char_hidden_size)
            char_p = tf.nn.embedding_lookup(self.char_emb, char_p)
            char_h = tf.nn.embedding_lookup(self.char_emb, char_h)
            
            _, (char_p, _) = tf.nn.dynamic_rnn(self.char_LSTM, char_p, dtype = tf.float32)
            _, (char_h, _) = tf.nn.dynamic_rnn(self.char_LSTM, char_h, dtype = tf.float32)
            
            # (batch, seq_len, char_hidden_size)
            char_p = tf.reshape(char_p, [-1, seq_len_p, args.char_hidden_size])
            char_h = tf.reshape(char_h, [-1, seq_len_h, args.char_hidden_size])

            # (batch, seq_len, word_dim + char_hidden_size)
            p = tf.concat([p, char_p], axis = -1)
            h = tf.concat([h, char_h], axis = -1)

        p = dropout(p)
        h = dropout(h)
        
        # ----- Context Representation Layer -----
        with tf.variable_scope("context_representation", reuse = tf.AUTO_REUSE):
            # (batch, seq_len, hidden_size * 2)
            con_p, _ = tf.nn.bidirectional_dynamic_rnn(
                self.context_LSTM_fw,
                self.context_LSTM_bw,
                p,
                dtype = tf.float32
            )
            con_p = tf.concat(con_p, axis = -1)

            con_h, _ = tf.nn.bidirectional_dynamic_rnn(
                self.context_LSTM_fw,
                self.context_LSTM_bw,
                h,
                dtype = tf.float32
            )
            con_h = tf.concat(con_p, axis = -1)

        con_p = dropout(con_p)
        con_h = dropout(con_h)

        # (batch, seq_len, hidden_size)
        con_p_fw, con_p_bw = tf.split(con_p, num_or_size_splits = 2, axis = -1)
        con_h_fw, con_h_bw = tf.split(con_h, num_or_size_splits = 2, axis = -1)

        # 1. Full-Matching

        # (batch, seq_len, hidden_size), (batch, hidden_size)
        # -> (batch, seq_len, l)
        mv_p_full_fw = mp_matching_func(con_p_fw, con_h_fw[:, -1, :], self.mp_w1)
        mv_p_full_bw = mp_matching_func(con_p_bw, con_h_bw[:, 0, :], self.mp_w2)
        mv_h_full_fw = mp_matching_func(con_h_fw, con_p_fw[:, -1, :], self.mp_w1)
        mv_h_full_bw = mp_matching_func(con_h_bw, con_p_bw[:, 0, :], self.mp_w2)

        # 2. Maxpooling-Matching
        # (batch, seq_len1, seq_len2, l)
        mv_max_fw = mp_matching_func_pairwise(con_p_fw, con_h_fw, self.mp_w3)
        mv_max_bw = mp_matching_func_pairwise(con_p_bw, con_h_bw, self.mp_w4)

        # (batch, seq_len, l)
        mv_p_max_fw = tf.reduce_max(mv_max_fw, axis = 2)
        mv_p_max_bw = tf.reduce_max(mv_max_bw, axis = 2)
        mv_h_max_fw = tf.reduce_max(mv_max_fw, axis = 1)
        mv_h_max_bw = tf.reduce_max(mv_max_bw, axis = 1)
        
        # 3. Attentive-Matching

        # (batch, seq_len1, seq_len2)
        att_fw = attention(con_p_fw, con_h_fw)
        att_bw = attention(con_p_bw, con_h_bw)


        # (batch, seq_len2, hidden_size) -> (batch, 1, seq_len2, hidden_size)
        # (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
        # -> (batch, seq_len1, seq_len2, hidden_size)
        att_h_fw = tf.expand_dims(con_h_fw, axis = 1) * tf.expand_dims(att_fw, axis = 3)
        att_h_bw = tf.expand_dims(con_h_bw, axis = 1) * tf.expand_dims(att_bw, axis = 3)

        # (batch, seq_len1, hidden_size) -> (batch, seq_len1, 1, hidden_size)
        # (batch, seq_len1, seq_len2) -> (batch, seq_len1, seq_len2, 1)
        # -> (batch, seq_len1, seq_len2, hidden_size)
        att_p_fw = tf.expand_dims(con_p_fw, axis = 2) * tf.expand_dims(att_fw, axis = 3)
        att_p_bw = tf.expand_dims(con_p_bw, axis = 2) * tf.expand_dims(att_bw, axis = 3)

        # (batch, seq_len1, hidden_size) / (batch, seq_len1, 1) -> (batch, seq_len1, hidden_size)
        att_mean_h_fw = div_with_small_value(tf.reduce_sum(att_h_fw, axis = 2), tf.reduce_sum(att_fw, axis = 2, keepdims = True))
        att_mean_h_bw = div_with_small_value(tf.reduce_sum(att_h_bw, axis = 2), tf.reduce_sum(att_bw, axis = 2, keepdims = True))

        # (batch, seq_len2, hidden_size) / (batch, seq_len2, 1) -> (batch, seq_len2, hidden_size)
        att_mean_p_fw = div_with_small_value(tf.reduce_sum(att_p_fw, axis = 1), tf.transpose(tf.reduce_sum(att_fw, axis = 1, keepdims = True), [0,2,1]))
        att_mean_p_bw = div_with_small_value(tf.reduce_sum(att_p_bw, axis = 1), tf.transpose(tf.reduce_sum(att_bw, axis = 1, keepdims = True), [0,2,1]))


        # (batch, seq_len, l)
        mv_p_att_mean_fw = mp_matching_func(con_p_fw, att_mean_h_fw, self.mp_w5)
        mv_p_att_mean_bw = mp_matching_func(con_p_bw, att_mean_h_bw, self.mp_w6)
        mv_h_att_mean_fw = mp_matching_func(con_h_fw, att_mean_p_fw, self.mp_w5)
        mv_h_att_mean_bw = mp_matching_func(con_h_bw, att_mean_p_bw, self.mp_w6)

        

        # 4. Max-Attentive-Matching

        # (batch, seq_len1, hidden_size)
        att_max_h_fw = tf.reduce_max(att_h_fw, axis = 2)
        att_max_h_bw = tf.reduce_max(att_h_bw, axis = 2)
        # (batch, seq_len2, hidden_size)
        att_max_p_fw = tf.reduce_max(att_p_fw, axis = 1)
        att_max_p_bw = tf.reduce_max(att_p_bw, axis = 1)

        # (batch, seq_len, l)
        mv_p_att_max_fw = mp_matching_func(con_p_fw, att_max_h_fw, self.mp_w7)
        mv_p_att_max_bw = mp_matching_func(con_p_bw, att_max_h_bw, self.mp_w8)
        mv_h_att_max_fw = mp_matching_func(con_h_fw, att_max_p_fw, self.mp_w7)
        mv_h_att_max_bw = mp_matching_func(con_h_bw, att_max_p_bw, self.mp_w8)

        # print("mv_h_att_max_bw = ", mv_h_att_max_bw.shape.as_list())
        # (batch, seq_len, l * 8)
        mv_p = tf.concat(
            [mv_p_full_fw, mv_p_max_fw, mv_p_att_mean_fw, mv_p_att_max_fw,
            mv_p_full_bw, mv_p_max_bw, mv_p_att_mean_bw, mv_p_att_max_bw], axis = 2
        )
        
        mv_h = tf.concat(
            [mv_h_full_fw, mv_h_max_fw, mv_h_att_mean_fw, mv_h_att_max_fw,
            mv_h_full_bw, mv_h_max_bw, mv_h_att_mean_bw, mv_h_att_max_bw], axis = 2
        )
        
        mv_p = dropout(mv_p)
        mv_h = dropout(mv_h)
        
        # ----- Aggregation Layer -----
        with tf.variable_scope("aggregation_layer", reuse = tf.AUTO_REUSE):
            # (batch, seq_len, l * 8) -> (batch, 2 * hidden_size)
            _, p_states = tf.nn.bidirectional_dynamic_rnn(
                self.aggregation_LSTM_fw,
                self.aggregation_LSTM_bw,
                mv_p,
                dtype = tf.float32
            )
            _, h_states = tf.nn.bidirectional_dynamic_rnn(
                self.aggregation_LSTM_fw,
                self.aggregation_LSTM_bw,
                mv_h,
                dtype = tf.float32
            )

            x = tf.concat([p_states[0][0], p_states[1][0], h_states[0][0], h_states[1][0]], axis = -1)
        x = dropout(x)
        
        # ----- Prediction Layer -----
        x = self.pred_fc1(x)
        x = dropout(x)
        self.logits = self.pred_fc2(x)
        self.prediction = tf.argmax(self.logits, axis = -1)


        if self.is_train:
            self._create_training_tensors()
        

    def train(self, session, save_path, train_data, valid_data,
            batch_size, epochs):
        
        saver = tf.train.Saver(self.get_trainable_variables(), max_to_keep=1)
        writer = tf.summary.FileWriter(save_path, session.graph)

        for epoch in range(epochs):
            
            train_loss = 0.0
            train_prediction = np.array([])

            val_loss = 0.0
            val_prediction = np.array([])

            for p,h,y in data_generator(batch_size, train_data):
                num = p.shape[0]
                feed_dict = {self.p: p, self.h: h, self.y: y, self.dropout_rate: self.args["dropout"]}
                _, cost, prediction, global_step = session.run([self.train_op, self.loss, self.prediction, self.global_step], feed_dict)
                train_loss += cost * num 
                train_prediction = np.append(train_prediction, prediction)

            train_loss = train_loss / train_data[0].shape[0]

            y_pred = train_prediction.flatten()
            y_true = np.array(train_data[2])

            train_acc = accuracy_score(y_true, y_pred)
            if self.args["class_size"] == 2:
                train_recall = recall_score(y_true, y_pred)
                train_precision = precision_score(y_true, y_pred)
                train_f1_score = f1_score(y_true, y_pred)
            else:
                train_recall = recall_score(y_true, y_pred, average = "micro")
                train_precision = precision_score(y_true, y_pred, average = "micro")
                train_f1_score = f1_score(y_true, y_pred, average = "micro")

            for p,h,y in data_generator(batch_size, valid_data):
                num = p.shape[0]
                feed_dict = {self.p: p, self.h: h, self.y: y, self.dropout_rate: 0.0}
                cost, prediction, global_step = session.run([self.loss, self.prediction, self.global_step], feed_dict)
                val_loss += cost * num 
                val_prediction = np.append(val_prediction, prediction)

            val_loss = val_loss / valid_data[0].shape[0]

            y_pred = val_prediction.flatten()
            y_true = np.array(valid_data[2])

            val_acc = accuracy_score(y_true, y_pred)
            if self.args["class_size"] == 2:
                val_recall = recall_score(y_true, y_pred)
                val_precision = precision_score(y_true, y_pred)
                val_f1_score = f1_score(y_true, y_pred)
            else:
                val_recall = recall_score(y_true, y_pred, average = "micro")
                val_precision = precision_score(y_true, y_pred, average = "micro")
                val_f1_score = f1_score(y_true, y_pred, average = "micro")



            train_loss_summary = tf.Summary(value = [tf.Summary.Value(tag = "train_loss", simple_value=train_loss)])
            writer.add_summary(train_loss_summary, global_step)

            train_acc_summary = tf.Summary(value = [tf.Summary.Value(tag = "train_acc", simple_value=train_acc)])
            writer.add_summary(train_acc_summary, global_step)

            train_recall_summary = tf.Summary(value = [tf.Summary.Value(tag = "train_recall", simple_value=train_recall)])
            writer.add_summary(train_recall_summary, global_step)

            train_precision_summary = tf.Summary(value = [tf.Summary.Value(tag = "train_precision", simple_value=train_precision)])
            writer.add_summary(train_precision_summary, global_step)
            
            train_f1_summary = tf.Summary(value = [tf.Summary.Value(tag = "train_f1", simple_value=train_f1_score)])
            writer.add_summary(train_f1_summary, global_step)


            val_loss_summary = tf.Summary(value = [tf.Summary.Value(tag = "val_loss", simple_value=val_loss)])
            writer.add_summary(val_loss_summary, global_step)


            val_acc_summary = tf.Summary(value = [tf.Summary.Value(tag = "val_accc", simple_value=val_acc)])
            writer.add_summary(val_acc_summary, global_step)


            val_recall_summary = tf.Summary(value = [tf.Summary.Value(tag = "val_recall", simple_value=val_recall)])
            writer.add_summary(val_recall_summary, global_step)

            val_precision_summary = tf.Summary(value = [tf.Summary.Value(tag = "val_precision", simple_value=val_precision)])
            writer.add_summary(val_precision_summary, global_step)
            
            val_f1_summary = tf.Summary(value = [tf.Summary.Value(tag = "val_f1", simple_value=val_f1_score)])
            writer.add_summary(val_f1_summary, global_step)



            print("Epoch : {0}, train_loss: {1}, train_acc: {2}, val_loss: {3}, val_acc : {4}".format(epoch + 1, train_loss, train_acc, val_loss, val_acc))

            

            self.save(saver, session, save_path)
        


    def predict(self, session, directory, data):

        feed_dict = {self.p: data[0], self.h: data[1], self.dropout_rate: 0.0}
        prediction = session.run([self.prediction], feed_dict)[0]

        path = os.path.join(directory, "prediction")

        np.save(path, prediction)



    def save(self, saver, session, directory):
        """
        Save the autoencoder model and metadata to the specified
        directory.
        """
        super(BIMPM, self).save(saver, session, directory)
        metadata = {'word_vocab_size': self.args["word_vocab_size"],
                    'word_dim': self.args["word_dim"],
                    'hidden_size': self.args["hidden_size"],
                    'num_perspective': self.args["num_perspective"],
                    "class_size": self.args["class_size"]
                    }
        metadata["use_char_emb"] = self.args["use_char_emb"]
        metadata["char_hidden_size"] = self.args["char_hidden_size"]
        metadata["char_dim"] = self.args["char_dim"]
        metadata["max_word_len"] = self.args["max_word_len"]
        metadata["char_vocab_size"] = self.args["char_vocab_size"]

        metadata_path = os.path.join(directory, 'metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f)
    
    @classmethod
    def load(cls, directory, session):
        """
        Load an instance of this class from a previously saved one.
        :param directory: directory with the model files
        :param session: tensorflow session
        :return: a BIMPM instance
        """
        model_path = os.path.join(directory, 'model')
        metadata_path = os.path.join(directory, 'metadata.json')
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        model = BIMPM(metadata)
        vars_to_load = model.get_trainable_variables()
        saver = tf.train.Saver(vars_to_load)
        saver.restore(session, model_path)
        return model
    
