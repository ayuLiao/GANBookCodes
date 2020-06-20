import tensorflow as tf
from story import Story
import os
import numpy as np


image_summary = tf.summary.image
scalar_summary = tf.summary.scalar
histogram_summary = tf.summary.histogram
merge_summary = tf.summary.merge
SummaryWriter = tf.summary.FileWriter

class Model(object):
    def __init__(self):
        self.story = Story()
        # 初始的句子向量
        self.vocab = self.story.vocab
        self.batch_size = self.story.batch_size - 2 #126
        self.chunk_size = self.story.chunk_size
        self.embedding_dim = 300
        self.num_units = 500
        self.learning_rate = 0.001
        self.epoch = 25
        self.sample_size = 50

    def gru_encoder(self, encode_emb, length, train=True):
        batch_size = self.batch_size if train else 1
        with tf.variable_scope('encoder'):
            cell = tf.nn.rnn_cell.GRUCell(num_units=self.num_units)
            inital_state = cell.zero_state(batch_size, tf.float32)
            _, final_state = tf.nn.dynamic_rnn(cell, encode_emb,
                                               initial_state=inital_state, sequence_length=length)
        return inital_state, final_state

    def softmax_variable(self,num_units, vocab_size, reuse=False):
        with tf.variable_scope('softmax_variable', reuse=reuse):
            w = tf.get_variable('w', [num_units, vocab_size])
            b = tf.get_variable('b', [vocab_size])
        return w, b

    def gru_decoder(self,decode_emb, length, state,scope, reuse=False):
        with tf.variable_scope(scope):
            cell = tf.nn.rnn_cell.GRUCell(num_units=self.num_units)
            outputs, final_state = tf.nn.dynamic_rnn(cell, decode_emb,
                                                 initial_state=state, sequence_length=length)
        x = tf.reshape(outputs, [-1, self.num_units])
        w, b = self.softmax_variable(self.num_units, len(self.vocab), reuse=reuse)
        logits = tf.matmul(x,w) + b
        prediction = tf.nn.softmax(logits, name='redictions')
        return logits, prediction, final_state

    def _loss(self, logits, targets, scope='loss'):
        with tf.variable_scope(scope):
            y_one_hot = tf.one_hot(targets, len(self.vocab))
            y_reshaped = tf.reshape(y_one_hot, [-1, len(self.vocab)])
            loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_reshaped))
        return loss



    def _optimizer(self, loss, scope='optimizer'):
        with tf.variable_scope(scope):
            grad_clip = 5
            tvars = tf.trainable_variables()
            grads,_ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
            op = tf.train.AdamOptimizer(self.learning_rate)
            optimizer = op.apply_gradients(zip(grads, tvars))
        return optimizer



    def _inputs(self):
        with tf.variable_scope('inputs'):
            self.encode_length = tf.placeholder(tf.int32, shape=[None, ], name='encode_length')
            self.decode_pre_x = tf.placeholder(tf.int32, shape=[None, None], name='decode_pre_x')
            self.decode_pre_y = tf.placeholder(tf.int32, shape=[None, None], name='decode_pre_y')
            self.decode_pre_length = tf.placeholder(tf.int32, shape=[None, ], name='decode_pre_length')
            self.decode_post_x = tf.placeholder(tf.int32, shape=[None, None], name='decode_post_x')
            self.decode_post_y = tf.placeholder(tf.int32, shape=[None, None], name='decode_post_y')
            self.decode_post_length = tf.placeholder(tf.int32, shape=[None, ], name='decode_post_length')

    def _embedding(self):
        with tf.variable_scope('embedding'):
            self.embedding = tf.get_variable(name='embedding', shape=[len(self.vocab), self.embedding_dim],
                                        initializer=tf.random_uniform_initializer(-0.1, 0.1))
            self.encode_emb = tf.nn.embedding_lookup(self.embedding, self.encode, name='encode_emb')
            self.decode_pre_emb = tf.nn.embedding_lookup(self.embedding, self.decode_pre_x, name='decode_pre_emb')
            self.decode_post_emb = tf.nn.embedding_lookup(self.embedding, self.decode_post_x, name='decode_post_emb')

    def build_model(self):
        self._inputs()
        self._embedding()
        self.initial_state, self.final_state =self.gru_encoder(self.encode_emb, self.encode_length)
        self.pre_logits, self.pre_prediction, self.pre_state = self.gru_decoder(self.decode_pre_emb,
                                                                                self.decode_pre_length, self.final_state,
                                                                                scope='decoder_pre')
        self.post_logits, self.post_prediction, self.post_state = self.gru_decoder(self.decode_post_emb,
                                                                                   self.decode_post_length, self.final_state,
                                                                      scope='decoder_post', reuse=True)

        self.pre_loss = self._loss(self.pre_logits, self.decode_pre_y, scope='decoder_pre_loss')
        self.pre_loss_sum = scalar_summary("pre_loss", self.pre_loss)
        self.post_loss = self._loss(self.post_logits, self.decode_post_y, scope='decoder_post_loss')
        self.post_loss_sum = scalar_summary("post_loss", self.post_loss)

        self.pre_optimizer = self._optimizer(self.pre_loss, scope='decoder_pre_op')
        self.post_optimizer = self._optimizer(self.post_loss, scope='decoder_post_op')


    def train(self):
        model_path = './output/skipThought.model'
        self.build_model()
        self.saver = tf.train.Saver()

        with tf.Session() as sess:
            self.writer = SummaryWriter("./output/logs", sess.graph)
            self._sum = merge_summary(
                [ self.pre_loss_sum, self.post_loss_sum])
            step = 0
            sess.run(tf.global_variables_initializer())
            new_state = sess.run(self.initial_state)

            for epoch in range(self.epoch):
                batches = self.story.batch()
                for encode_x, decode_pre_x, decode_pre_y, \
                    decode_post_x, decode_post_y, encode_length, \
                    decode_pre_length, decode_post_length in batches:
                    if len(encode_x) != self.batch_size: continue
                    feed = {
                        self.initial_state:new_state,
                        self.encode: encode_x,
                        self.encode_length: encode_length,
                        self.decode_pre_x: decode_pre_x,
                        self.decode_pre_y: decode_pre_y,
                        self.decode_pre_length: decode_pre_length,
                        self.decode_post_x: decode_post_x,
                        self.decode_post_y: decode_post_y,
                        self.decode_post_length: decode_post_length
                    }
                    _, pre_loss, _, _, post_loss, new_state,summary_str = sess.run(
                        [self.pre_optimizer, self.pre_loss, self.pre_state,
                         self.post_optimizer, self.post_loss, self.post_state,self._sum],
                        feed_dict=feed)
                    self.writer.add_summary(summary_str, step)

                    print(' epoch:', epoch,
                          ' step:', step, ' pre_loss', pre_loss,
                          ' post_loss', post_loss)
                    step += 1
                self.saver.save(sess, model_path, global_step=step)



    def gen(self):
        self._inputs()

        self._embedding()
        self.initial_state, self.final_state = self.gru_encoder(self.encode_emb, self.encode_length,train=False)

        self.post_logits, self.post_prediction, self.post_state = self.gru_decoder(self.decode_post_emb,
                                                                                   self.decode_post_length,
                                                                                   self.final_state,
                                                                                   scope='decoder_post')
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            new_state = sess.run(self.initial_state)
            saver.restore(sess, tf.train.latest_checkpoint('./output/'))
            encode_x = [[self.story.word_to_int[c] for c in '宝玉归来家中']]
            samples = [[] for _ in range(self.sample_size)]
            samples[0] = encode_x[0]
            for i in range(self.sample_size):
                decode_x = [[self.story.word_to_int['<GO>']]]
                while decode_x[0][-1] != self.story.word_to_int['<EOS>']:
                    feed = {self.encode: encode_x, self.encode_length: [len(encode_x[0])], self.initial_state: new_state,
                            self.decode_post_x: decode_x, self.decode_post_length: [len(decode_x[0])]}
                    predict, state = sess.run([self.post_prediction, self.post_state], feed_dict=feed)
                    int_word = np.argmax(predict, 1)[-1]
                    decode_x[0] += [int_word]
                samples[i] += decode_x[0][1:-1]
                encode_x = [samples[i]]
                new_state = state
                print(''.join([self.story.int_to_word[sample] for sample in samples[i]]))



