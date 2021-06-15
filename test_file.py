import tensorflow as tf

# v1 = tf.Variable([1,2], dtype=tf.float32)
# x = 0.5
# y = [0.1, 0.2]
# print(v1*x + y)

t1 = tf.Variable(initial_value=1.0)
l1 = [0,1]

print(tf.Variable(t1.numpy() ,dtype=tf.float32))
print(t1)