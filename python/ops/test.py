import tensorflow as tf

#tf.enable_eager_execution()

import mp3_decode_op

mp3d = mp3_decode_op.mp3_decode

d = tf.io.read_file("../../sine440_cbr128.mp3")
md = mp3d(d)

print(md)
print(md.numpy().sum())
