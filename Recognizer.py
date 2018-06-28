import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.rnn as rnns


def layer_stack(in_map, n_layers, n_channels, last=False, is_training=True):
    convs = [in_map]
    for i in range(0, n_layers):
        conv = tf.nn.relu(
            layers.batch_norm(layers.conv2d(convs[-1], num_outputs=n_channels, kernel_size=3, activation_fn=None,
                                            stride=1, padding='SAME'), is_training=is_training, decay=0.95))
        convs.append(conv)

    if last:
        convs[-1] = layers.dropout(convs[-1], keep_prob=0.8, is_training=is_training)
    return convs[-1]


class MathFormulaRecognizer():
    def __init__(self, num_label, dim_hidden):
        # paprameters for the model
        self.x = tf.placeholder(tf.float32, [None, None, None, 1])
        self.x_mask = tf.placeholder(tf.float32, [None, None, None])
        self.ex_mask = tf.expand_dims(self.x_mask, 3)

        self.y = tf.placeholder(tf.int32, [None, None])
        self.y_mask = tf.placeholder(tf.float32, [None, None])
        self.seq_length = tf.shape(self.y)[1]

        self.initial_lr = 0.2
        self.num_label = num_label
        self.dim_hidden = dim_hidden
        self.coverage_depth = 128
        self.dim_embed = 128
        self.is_train = tf.placeholder(tf.bool)
        self.batch_size = tf.shape(self.x)[0]
        self.in_height = tf.shape(self.x)[1]
        self.in_width = tf.shape(self.x)[2]
        self.latent_depth = 128
        self.attention_dimension = self.latent_depth

        self.weight_initializer = tf.contrib.layers.xavier_initializer()
        self.bias_initializer = tf.constant_initializer(0.0)
        self.emb_initializer = tf.random_uniform_initializer(minval=-1.0, maxval=1.0)
        self.start_token = 111
        self.construct_encoder()
        self.set_Decoder_parameter()
        self.build_train()

    def project_features(self, features, name):
        with tf.variable_scope(name):
            if name == 'f':
                w = self.w_f
                b = self.bias_f
                in_depth = self.coverage_depth
            else:
                w = self.w_annotation
                b = self.bias_annotation
                in_depth = self.latent_depth
            features_flat = tf.reshape(features, [-1, in_depth])
            features_proj = tf.matmul(features_flat, w) + b
            features_proj = tf.reshape(features_proj, [-1, self.feature_size, self.attention_dimension])
            return features_proj

    def construct_encoder(self):
        with tf.variable_scope('Encoder'):
            first_stack = layer_stack(self.x, 3, 32, is_training=self.is_train)
            pooled = layers.max_pool2d(first_stack, 2, 2, padding='VALID')

            second_stack = layer_stack(pooled, 3, 64, is_training=self.is_train)
            pooled = layers.max_pool2d(second_stack, 2, 2, padding='VALID')

            third_stack = layer_stack(pooled, 3, 64, is_training=self.is_train)
            pooled = layers.max_pool2d(third_stack, 2, 2, padding='VALID')

            fourth_stack = layer_stack(pooled, 3, 128, True, self.is_train)
            pooled = layers.max_pool2d(fourth_stack, 2, 2, padding='VALID')

            self.ex_mask = layers.max_pool2d(self.ex_mask, 2, 2, padding='VALID')
            self.ex_mask = layers.max_pool2d(self.ex_mask, 2, 2, padding='VALID')
            self.ex_mask = layers.max_pool2d(self.ex_mask, 2, 2, padding='VALID')
            self.ex_mask = layers.max_pool2d(self.ex_mask, 2, 2, padding='VALID')

            self.information_tensor = pooled

    def set_Decoder_parameter(self):
        # Decoder:
        self.feature_height = tf.shape(self.information_tensor)[1]
        self.feature_width = tf.shape(self.information_tensor)[2]
        self.feature_size = self.feature_height * self.feature_width
        self.mean_feature = tf.reduce_mean(self.information_tensor, axis=[1, 2])
        self.vec_mask = tf.reshape(self.ex_mask, [-1, self.feature_size])

        with tf.variable_scope('Decoder'):
            # notice that for gru, out equal to state
            self.gru = rnns.GRUCell(self.dim_hidden)

            self.w_hidden = tf.get_variable("w_hidden", shape=[self.dim_hidden, self.attention_dimension],
                                            initializer=self.weight_initializer)

            self.bias_hidden = tf.get_variable("bias_hidden", shape=[self.attention_dimension],
                                               initializer=self.bias_initializer)

            self.w_annotation = tf.get_variable("w_annotation", shape=[self.latent_depth, self.attention_dimension],
                                                initializer=self.weight_initializer)

            self.bias_annotation = tf.get_variable("bias_annotation", shape=[self.attention_dimension],
                                                   initializer=self.bias_initializer)

            self.w_f = tf.get_variable("w_f", shape=[self.coverage_depth, self.attention_dimension],
                                       initializer=self.weight_initializer)

            self.bias_f = tf.get_variable("bias_f", shape=[self.attention_dimension],
                                          initializer=self.bias_initializer)

            self.w_2e = tf.get_variable("w_2e", shape=[self.attention_dimension, 1],
                                        initializer=self.weight_initializer)

            self.bias_2e = tf.get_variable("bias_2e", shape=[1],
                                           initializer=self.bias_initializer)

            self.w_init2hid = tf.get_variable('w_init2hid', shape=[self.latent_depth, self.dim_hidden],
                                              initializer=self.weight_initializer)

            self.bias_init2hid = tf.get_variable("bias_init2hid", shape=[self.dim_hidden],
                                                 initializer=self.bias_initializer)

            self.w_B2f_filter = tf.get_variable("w_B2f_filter", shape=[11, 11, 1, self.coverage_depth],
                                                initializer=self.weight_initializer)

            self.b_B2f = tf.get_variable("b_B2f", shape=[self.coverage_depth], initializer=self.bias_initializer)

            self.w_2logit = tf.get_variable('w_2logit', shape=[self.dim_hidden, self.num_label],
                                            initializer=self.weight_initializer)

            self.bias_2logit = tf.get_variable("bias_2logit", shape=[self.num_label],
                                               initializer=self.bias_initializer)

            self.w_embedding = tf.get_variable('w_embedding', shape=[self.num_label, self.dim_embed],
                                               initializer=self.emb_initializer)

            self.bias_embedding = tf.get_variable("bias_embedding", shape=[self.dim_embed],
                                                  initializer=self.bias_initializer)

            self.counter_dis = tf.Variable(trainable=False, initial_value=0, dtype=tf.int32)

    def decoding_one_word_train(self, beta_t, state, previous_vec, i):
        F = tf.nn.conv2d(tf.reshape(beta_t, [-1, self.feature_height, self.feature_width, 1]), self.w_B2f_filter,
                         strides=[1, 1, 1, 1], padding='SAME')
        F = tf.nn.bias_add(F, self.b_B2f)

        # though it's okay to directly use state, in this way it is clearer.
        out = state
        weighted_h = tf.matmul(out, self.w_hidden) + self.bias_hidden

        weighted_annotation = self.project_features(self.information_tensor, 'feature')
        weighted_f = self.project_features(F, 'F')
        SUM = tf.add(tf.add(tf.expand_dims(weighted_h, 1), weighted_annotation),
                     weighted_f)  # (batch_size,feature_size,attention_dimension)
        e = tf.matmul(tf.nn.tanh(tf.reshape(SUM, [-1, self.attention_dimension])),
                      self.w_2e) + self.bias_2e  # (batch_size*feature_size)
        e = tf.reshape(e, [-1, self.feature_size])  # (batch_size,feature_size)

        alpha_t = tf.exp(e)  # (batch_size,feature_size)
        alpha_t = tf.multiply(alpha_t, self.vec_mask)  # mask out blanks.
        alpha_t = alpha_t / tf.expand_dims(tf.reduce_sum(alpha_t, axis=-1), 1)
        beta_t = beta_t + alpha_t

        c = tf.reduce_sum(tf.multiply(
            tf.transpose(tf.reshape(self.information_tensor, [-1, self.feature_size, self.latent_depth]), [2, 0, 1]),
            alpha_t), axis=-1)
        c = tf.transpose(c, [1, 0])

        # hard coding word vec to start token
        word_embedding = tf.nn.embedding_lookup(self.w_embedding, previous_vec) + self.bias_embedding
        gru_in = tf.concat([c, word_embedding], axis=1)
        out, state = self.gru(gru_in, state)

        labels = self.y[:, i]
        labels = tf.expand_dims(labels, 1)
        batch_range = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
        sparse = tf.concat([batch_range, labels], 1)
        onehot = tf.sparse_to_dense(sparse, tf.stack([self.batch_size, self.num_label]), 1.0, 0.0)
        logit = tf.matmul(out, self.w_2logit) + self.bias_2logit
        # make those self just make it easier to debug
        xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=logit, labels=onehot)
        xentropy = xentropy * self.y_mask[:, i]

        loss = tf.reduce_sum(xentropy)
        return loss, beta_t, out

    def decoding_one_word_validate(self, beta_t, state, previous_vec):
        F = tf.nn.conv2d(tf.reshape(beta_t, [-1, self.feature_height, self.feature_width, 1]), self.w_B2f_filter,
                         strides=[1, 1, 1, 1], padding='SAME')
        F = tf.nn.bias_add(F, self.b_B2f)

        # though it's okay to directly use state as out, in this way it is clearer.
        out = state
        weighted_h = tf.matmul(out, self.w_hidden) + self.bias_hidden

        weighted_annotation = self.project_features(F, 'feature')
        weighted_f = self.project_features(self.F, 'f')
        SUM = tf.add(tf.add(tf.expand_dims(weighted_h, 1), weighted_annotation),
                     weighted_f)  # (batch_size,feature_size,attention_dimension)
        e = tf.matmul(tf.nn.tanh(tf.reshape(SUM, [-1, self.attention_dimension])),
                      self.w_2e) + self.bias_2e  # (batch_size*feature_size)
        e = tf.reshape(e, [-1, self.feature_size])  # (batch_size,feature_size)

        alpha_t = tf.exp(e)  # (batch_size,feature_size)
        alpha_t = tf.multiply(alpha_t, self.vec_mask)  # mask out blanks.
        alpha_t = alpha_t / tf.expand_dims(tf.reduce_sum(alpha_t, axis=-1), 1)
        beta_t = beta_t + alpha_t

        c = tf.reduce_sum(tf.multiply(
            tf.transpose(tf.reshape(self.information_tensor, [-1, self.feature_size, self.latent_depth]), [2, 0, 1]),
            alpha_t), axis=-1)
        c = tf.transpose(c, [1, 0])

        word_embedding = tf.nn.embedding_lookup(self.w_embedding, previous_vec) + self.bias_embedding
        gru_in = tf.concat([c, word_embedding], axis=1)
        out, state = self.gru(gru_in, state)
        logit = tf.matmul(out, self.w_2logit) + self.bias_2logit
        return beta_t, out, logit, alpha_t

    def build_train(self):
        # initialization:
        beta_t = tf.zeros([self.batch_size, self.feature_size], dtype=tf.float32)
        state = tf.matmul(self.mean_feature, self.w_init2hid) + self.bias_init2hid
        total_loss = tf.constant(0.0, dtype=tf.float32)
        with tf.variable_scope('Decoder'):
            start_vec = tf.tile(tf.constant([self.start_token, ]), [self.batch_size])
            loss, beta_t, out = self.decoding_one_word_train(beta_t, state, start_vec, 0)
            total_loss += loss
            # first_round
            tf.get_variable_scope().reuse_variables()
            i = tf.constant(1)
            while_condition = lambda N1, i, N2, N3: tf.less(i, self.seq_length)

            # keep alpha_t for debugging, may get rid of it later.
            # Notice that for gru state = out
            def body(total_loss, i, beta_t, state):
                loss, beta_t, out = self.decoding_one_word_train(beta_t, state, self.y[:, i - 1], i)
                total_loss += loss
                return [total_loss, tf.add(i, 1), beta_t, out]

            # do the loop:
            [total_loss, i, beta_t, out] = tf.while_loop(while_condition, body,
                                                         [total_loss, i, beta_t, out])
            total_loss = total_loss / tf.reduce_sum(self.y_mask)

        self.lr = tf.train.exponential_decay(self.initial_lr, self.counter_dis, 1500, 0.96, staircase = True)
        opt = layers.optimize_loss(loss=total_loss, learning_rate=self.lr,
                                   optimizer=tf.train.AdadeltaOptimizer,
                                   clip_gradients=100., global_step=self.counter_dis)

        return total_loss, opt

    def build_greedy_eval(self, max_len=200):
        beta_t = tf.zeros([self.batch_size, self.feature_size], dtype=tf.float32)

        # c = tf.matmul(self.mean_feature,self.w_init2c) + self.bias_init2c
        state = tf.matmul(self.mean_feature, self.w_init2hid) + self.bias_init2hid
        out = state
        previous_word = self.previous_word
        words = []
        alphas = []
        betas = []
        with tf.variable_scope('Decoder'):
            for i in range(0, max_len):
                # have alpha_t for debugging
                beta_t, out, logit, alpha_t = self.decoding_one_word_validate(beta_t, state, previous_word)
                previous_word = tf.argmax(logit, 1)
                state = out
                words.append(previous_word)
                alphas.append(alpha_t)
                betas.append(beta_t)
        return words, alphas, betas

    def build_eval(self):

        self.in_beta_t = tf.zeros([self.batch_size, self.feature_size], dtype=tf.float32)
        beta_t = self.in_beta_t

        # c = tf.matmul(self.mean_feature,self.w_init2c) + self.bias_init2c
        self.in_state = tf.matmul(self.mean_feature, self.w_init2hid) + self.bias_init2hid
        state = self.in_state

        self.in_previous_word = tf.tile(tf.constant([111, ]), [self.batch_size])
        previous_word = self.in_previous_word

        with tf.variable_scope('Decoder'):
            beta_t, out, logit, alpha_t = self.decoding_one_word_validate(beta_t, state, previous_word)
            state = out
        return alpha_t, beta_t, state, logit
