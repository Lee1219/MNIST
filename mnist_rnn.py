import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

mnist = input_data.read_data_sets("./mnist/data", one_hot=True)
#mnist = keras.datasets.mnist
#(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
###### 옵션 ######
#학습률
learning_rate = 0.001
#학습 횟수
total_epoch = 5
#batch의 크기
batch_size = 128
#RNN에서 단계 별 입력되는 데이터의 개수
n_input = 28
#단계 수
n_step = 28
#은닉층 수
n_hidden = 128
#분류하고자 하는 카테고리 수, 출력층의 노드수
n_class = 10
#################

# X : 입력층을 의미 / placeholder를 통해 float32자료형을 가지고 () 모양의 텐서를 만듬
X = tf.placeholder(tf.float32, [None, n_step, n_input])
# Y : 출력층을 의미 / placeholder를 통해 텐서를 만듬
Y = tf.placeholder(tf.float32, [None, n_class])
# W : 가중치를 의미 / 은닉층에서 출력층으로 이어지는 가중치를 W로 정의
W = tf.Variable(tf.random_normal([n_hidden, n_class]))
b = tf.Variable(tf.random_normal([n_class]))

cell = tf.nn.rnn_cell.BasicRNNCell(n_hidden)

outputs, states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
#outputs: [batch_size, n_step, n_hidden]
# -> [n_step, batch_size, n_hidden]
outputs = tf.transpose(outputs, [1, 0, 2])

#-> [batch_size, n_hidden]
outputs = outputs[-1]

#y=x*W+b / matmul : 행렬 곱셈
model = tf.matmul(outputs, W) + b

#reduce_XXX : 텐서의 차원을 줄인다. softmax_cross_entropy_with_logits 함수를 통해서 손실값을 얻고 그 값을 평균을 내서 차원을 줄인다.

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
    logits = model, labels = Y))

#Adam 방식을 통해서 손실을 최소화하는 방식으로 최적화를 진행한다.
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

###############
#신경망 모델 학습
###############

sess = tf.Session()
sess.run(tf.global_variables_initializer())

#total_batch : epoch 동안에 몇 번의 batch가 들어가 있는지 수를 담고 있음.
#ex. 12개의 학습 데이터 batch가 4라면 total_batch는 3
total_batch = int(mnist.train.num_examples/batch_size)

for epoch in range(total_epoch):
    total_cost = 0

    for i in range(total_batch):
        #MNIST의 손글씨 데이터를 batch 수에 맞추어서 가져온다.
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape((batch_size, n_step, n_input))

        _, cost_val = sess.run([optimizer, cost], feed_dict={X: batch_xs, Y:batch_ys})

        total_cost += cost_val

    print("Epoch: ", '%4d' % (epoch +1), 'Avg. cost = ', "{:3f}".format(total_cost/total_batch))
print("최적화 완료")

###############
#결과 확인
###############

is_correct = tf.equal(tf.argmax(model, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
test_batch_size = len(mnist.test.images)
print(test_batch_size)
#print(mnist.test.images)
test_xs = mnist.test.images.reshape(test_batch_size, n_step, n_input)
#print(test_xs)
test_ys = mnist.test.labels
print("정확도 : ", sess.run(accuracy, feed_dict={X: test_xs, Y:test_ys}))
