import tensorflow as tf

from decode_mp3_op import decode_mp3


class DecodeMp3Test(tf.test.TestCase):

    def test_decode_mp3_normal(self):
        with self.session():
            data = tf.io.read_file("../../testfiles/sine440_cbr128.mp3")

            samples, sr = decode_mp3(data)

            self.assertAllEqual(samples.shape, [1, 44100])
            self.assertEqual(sr, 44100)

    def test_decode_mp3_mono_to_stereo(self):
        with self.session():
            data = tf.io.read_file("../../testfiles/sine440_cbr128.mp3")

            samples, sr = decode_mp3(data, desired_channels=2)

            self.assertAllEqual(samples.shape, [2, 44100])
            # mono channel should be duplicated
            self.assertAllEqual(samples[0], samples[1])

    def test_decode_mp3_cut_samples(self):
        with self.session():
            data = tf.io.read_file("../../testfiles/sine440_cbr128.mp3")

            samples_original, _ = decode_mp3(data)
            samples_cut, _ = decode_mp3(data, desired_samples=44000)

            self.assertAllEqual(samples_cut.shape, [1, 44000])
            # existing samples should be completely decoded
            self.assertAllEqual(samples_cut, samples_original[:, :44000])

    def test_decode_mp3_pad_samples(self):
        with self.session():
            data = tf.io.read_file("../../testfiles/sine440_cbr128.mp3")

            samples_original, _ = decode_mp3(data)
            samples_padded, _ = decode_mp3(data, desired_samples=44200)

            self.assertAllEqual(samples_padded.shape, [1, 44200])
            # existing samples should be completely decoded
            self.assertAllEqual(samples_padded[:,:44100], samples_original)
            # missing samples should be filled with zeros
            self.assertAllEqual(samples_padded[0,44100:], tf.zeros(100))


if __name__ == "__main__":
    tf.test.main()
