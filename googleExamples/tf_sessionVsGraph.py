# tutorial: https://danijar.com/what-is-a-tensorflow-session/
import tensorflow as tf


# graph = "setup the system"
graph = tf.Graph()
with graph.as_default():
    variable = tf.Variable(42, name='foo')
    initialize = tf.global_variables_initializer() # deprecated: tf.initialize_all_variables()
    assign = variable.assign(13)


# session = "run the system" (and allocate memory to do so too)
with tf.Session(graph=graph) as sess:
    sess.run(initialize)
    sess.run(assign)
    print(sess.run(variable))
# Output: 13


# this shows that variables are "local" to a session:
try:
    with tf.Session(graph=graph) as sess:
        print(sess.run(variable))
except:
    print('Error: Attempting to use uninitialized value foo')
# Error: Attempting to use uninitialized value foo


# this shows you can reuse the same graph in a different session IF you run the initialize(r)
with tf.Session(graph=graph) as sess:
    sess.run(initialize)
    # sess.run(assign)
    print(sess.run(variable))
# Output: 42
