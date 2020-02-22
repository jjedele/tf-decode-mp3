#include "tensorflow/core/framework/op_kernel.h"
#include <algorithm>
#include <array>
#include <iterator>
#include <vector>

#define MINIMP3_IMPLEMENTATION
#include "../minimp3/minimp3.h"

using namespace tensorflow;

class Mp3DecodeOp : public OpKernel {
public:
  explicit Mp3DecodeOp(OpKernelConstruction *context) : OpKernel(context) {}

  void Compute(OpKernelContext *context) override {
    // get the input data, i.e. encoded mp3 data
    const Tensor &input_tensor = context->input(0);
    const string &input_data = input_tensor.scalar<tstring>()();

    // initialize mp3 decoder
    static mp3dec_t mp3d;
    mp3dec_init(&mp3d);

    // decode mp3
    mp3dec_frame_info_t info;
    std::vector<short> decoded_data;
    std::array<short, MINIMP3_MAX_SAMPLES_PER_FRAME> buffer;
    int samples;
    const uint8_t *data_ptr = (uint8_t *)input_data.data();
    int data_left = input_data.size();

    int n_samples = 0;
    while (data_left > 0 &&
           (samples = mp3dec_decode_frame(&mp3d, data_ptr, data_left,
                                          buffer.data(), &info))) {
      decoded_data.insert(decoded_data.end(), buffer.begin(), buffer.begin() + samples);
      n_samples += samples;
      data_ptr += info.frame_bytes;
      data_left -= info.frame_bytes;
    }

    // create the output tensor
    Tensor *output_tensor = NULL;
    TensorShape output_shape;
    output_shape.AddDim(n_samples);
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &output_tensor));
    auto output_flat = output_tensor->flat<int16>();

    // copy samples into output buffer
    // TODO: see if we can make this more efficient
    for (int i = 0; i < decoded_data.size(); i++) {
      output_flat(i) = decoded_data[i];
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("Mp3Decode").Device(DEVICE_CPU), Mp3DecodeOp);
