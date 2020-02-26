# TensorFlow DecodeMp3 Operator

An operator to decode MP3 data from in-memory data similar to [tf.audio.decode_wav](https://www.tensorflow.org/api_docs/python/tf/audio/decode_wav).
This is handy to decode large numbers of small MP3 files, e.g. when reading from a TFRecord file.
It uses [@lieff](https://github.com/lieff)'s awesome [minimp3](https://github.com/lieff/minimp3) library.

**Warning**: Not tested with stereo files yet.

## Installation

Only local installation supported so far.

	cd tf-decode-mp3
	make
	pip install -e .

## Usage

	import tensorflow as tf
	from tf_decode_mp3 import decode_mp3

	ds = (tf.data.Dataset
	        .list_files("*.mp3")
	        .map(tf.io.read_file)
	        .map(decode_mp3))

	for samples, sample_rate in ds:
	    print(samples.shape, sample_rate)


The `decode_mp3` function has two additional parameters:

* `desired_channels`: Fix how many channels are returned. If the file contains more, only the first will be used. If the file contains less, they will be duplicated.
* `desired_samples`: Fix how many samples are returned per channel. If the file contains more, the surplus is thrown away. If the file contains less, it will be padded with 0s on the right.


## Acknowledgements

* [minimp3](https://github.com/lieff/minimp3)
* [TF Custom Op Repo](https://github.com/tensorflow/custom-op)
