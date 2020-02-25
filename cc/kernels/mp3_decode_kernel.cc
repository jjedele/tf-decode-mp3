#include "tensorflow/core/framework/op_kernel.h"
#include <algorithm>
#include <array>
#include <iterator>
#include <vector>

#define MINIMP3_IMPLEMENTATION
#define MINIMP3_FLOAT_OUTPUT
#include "../minimp3/minimp3_ex.h"

using namespace tensorflow;

class Mp3DecodeOp : public OpKernel {
public:
  explicit Mp3DecodeOp(OpKernelConstruction *context) : OpKernel(context) {}

  void Compute(OpKernelContext *context) override {
    // get the input data, i.e. encoded mp3 data
    const Tensor &input_tensor = context->input(0);
    const string &input_data = input_tensor.scalar<tstring>()();

    // initialize mp3 decoder
    static mp3dec_t mp3dec;
    mp3dec_init(&mp3dec);

    // decode mp3
    static mp3dec_file_info_t mp3;
    mp3dec_load_buf(&mp3dec, (const uint8_t *)input_data.data(), input_data.size(),
                    &mp3, NULL /* progress callback */, NULL /* user data */);

    // create the output tensor
    Tensor *output_tensor = NULL;
    TensorShape output_shape;
    output_shape.AddDim(mp3.samples);
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &output_tensor));
    auto output_flat = output_tensor->flat<float>();

    // copy samples into output buffer
    // TODO: see if we can make this more efficient
    for (int i = 0; i < mp3.samples; i++) {
      output_flat(i) = mp3.buffer[i];
    }
  }
};

REGISTER_KERNEL_BUILDER(Name("Mp3Decode").Device(DEVICE_CPU), Mp3DecodeOp);
