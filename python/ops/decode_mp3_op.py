"""Decode MP3 operator."""

from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader

decode_mp3_op = load_library.load_op_library(
    resource_loader.get_path_to_datafile("_decode_mp3_op.so")
)
decode_mp3 = decode_mp3_op.decode_mp3
