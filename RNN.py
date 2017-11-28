import random
import numpy as np
def build_data(n):
    xs = []
    ys = []
    for i in range(2000):
        k = random.uniform(1,50)
        x = [[np.sin(k+j)] for j in range(0,n)]
        y = [np.sin(k + n)]
        xs.append(x)
        ys.append(y)
    train_x = np.array(xs[0:1500])
    train_y = np.array(ys[0:1500])
    test_x = np.array(xs[1500:])
    test_y = np.array(ys[1500:])
    return (train_x,train_y,test_x,test_y)
(train_x,train_y,test_x,test_y) = build_data(10)
print(train_x.shape,train_y.shape,test_x.shape,test_y.shape)
import tensorflow as tf
# from tensorflow.contrib.rnn.python.ops import core_rnn_cell
# from tensorflow.contrib.rnn.python.ops import core_rnn

length =10
time_step_size = length
vectore_size = 1
batch_size = 10
test_size = 10
X = tf.placeholder("float",[None,length,vectore_size])
Y = tf.placeholder("float",[None,1])    Z
W = tf.Variable(tf.random_normal([10,1],stddev=0.01))
B = tf.Variable(tf.random_normal([1],stddev = 0.01))
def seq_predict_model(X,w,b,time_step_size,vector_size):
     #输入的 Xshape [batch_size,time_step_size,vector_size]
     #transpose X to [time_step_size,batch_size,vecotr_size]
     X = tf.transpose(X,[1,0,2])
     X = tf.reshape(X,[-1,vector_size])
     X = tf.split(X,time_step_size,0)
     cell = tf.nn.rnn_cell.BasicRNNCell(num_units = 10)
     initial_state = cell.zero_state(batch_size,tf.float32)
     outputs,_states = tf.contrib.rnn.static_rnn(cell = cell,inputs = X,initial_state = initial_state)
     return tf.matmul(outputs[-1],w) + b,cell.state_size
pred_y,_ = seq_predict_model(X,W,B,time_step_size,vectore_size)
loss = tf.square(tf.subtract(Y,pred_y[-1]))
train_op = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    for i in range(50):
        for end in range(batch_size,len(train_x)):
            begin = end - batch_size
            x_value = train_x[begin:end]
            y_value = train_y[begin:end]
            sess.run(train_op,feed_dict={X:x_value,Y:y_value})
        test_indices = np.arange(len(test_x))
        np.random.shuffle(test_indices)
        test_indices = test_indices[0:test_size]
        x_value = test_x[test_indices]
        y_value = test_y[test_indices]
        val_loss = np.mean(sess.run(loss,feed_dict={X:x_value,Y:y_value}))
        print('RNN ,%s'% i,val_loss)










