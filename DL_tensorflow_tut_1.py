import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('/home/shanmukha/AnacondaProjects/Spyder_projects/mnist',
                                  one_hot=True )

nodes_hl_1 = nodes_hl_2 = nodes_hl_3 = 500
n_classes = 10
batch_size = 100

x = tf.placeholder('float',[None,784])
y = tf.placeholder('float')

def neural_network_model(data):
    hidden_layer_1 = {'weights':tf.Variable(tf.random_normal([784,nodes_hl_1])),
                      'biases' :tf.Variable(tf.random_normal([nodes_hl_1])) }
    hidden_layer_2 = {'weights':tf.Variable(tf.random_normal([nodes_hl_1,nodes_hl_2])),
                      'biases' :tf.Variable(tf.random_normal([nodes_hl_2])) }
    hidden_layer_3 = {'weights':tf.Variable(tf.random_normal([nodes_hl_2,nodes_hl_3])),
                      'biases' :tf.Variable(tf.random_normal([nodes_hl_3])) }
    output_layer = {'weights':tf.Variable(tf.random_normal([nodes_hl_3,n_classes])),
                      'biases' :tf.Variable(tf.random_normal([n_classes])) }
    
    l1 = tf.add(tf.matmul(data,hidden_layer_1['weights']),hidden_layer_1['biases'])
    l1 = tf.nn.relu(l1)
    l2 = tf.add(tf.matmul(l1,hidden_layer_2['weights']),hidden_layer_2['biases'])
    l2 = tf.nn.relu(l2)
    l3 = tf.add(tf.matmul(l2,hidden_layer_3['weights']),hidden_layer_3['biases'])
    l3 = tf.nn.relu(l3)
    output = tf.add(tf.matmul(l3,output_layer['weights']),output_layer['biases'])
    return output

def train_neural_network(x):
    prediction = neural_network_model(x)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(loss)
    tot_epochs = 10
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(tot_epochs):
            epoch_loss = 0
            for i in range(int(mnist.train.num_examples/batch_size)):
                epoch_x,epoch_y = mnist.train.next_batch(batch_size)
                j,c = sess.run([optimizer,loss],feed_dict={x:epoch_x,y:epoch_y})
                epoch_loss+=c
            print('Epoch ',epoch,' completed.Total loss = ',epoch_loss)
            correct = tf.equal(tf.argmax(prediction,1),tf.argmax(y,1))
            accuracy = tf.reduce_mean(tf.cast(correct,'float'))
            print('Accuracy: ',accuracy.eval({x:mnist.test.images,y:mnist.test.labels}))
train_neural_network(x)            
            
    