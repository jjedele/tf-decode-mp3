CXX := g++
PYTHON_BIN_PATH = python

SRCS = tf_decode_mp3/cc/ops/decode_mp3_op.cc tf_decode_mp3/cc/kernels/decode_mp3_kernel.cc

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11
LDFLAGS = -shared ${TF_LFLAGS}

TARGET_LIB = tf_decode_mp3/python/ops/_decode_mp3_op.so

decode_mp3_op: $(TARGET_LIB)

$(TARGET_LIB): $(SRCS)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}

clean:
	rm -f $(TARGET_LIB)
