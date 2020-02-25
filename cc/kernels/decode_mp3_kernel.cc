#include "tensorflow/core/framework/op_kernel.h"
#include <iostream>

#define MINIMP3_IMPLEMENTATION
#define MINIMP3_FLOAT_OUTPUT
#include "../minimp3/minimp3_ex.h"

using namespace tensorflow;

class DecodeMp3Op : public OpKernel {
public:
  explicit DecodeMp3Op(OpKernelConstruction *context) : OpKernel(context) {
    OP_REQUIRES_OK(context,
                   context->GetAttr("desired_channels", &desired_channels));
    OP_REQUIRES_OK(context,
                   context->GetAttr("desired_samples", &desired_samples));
  }

  void Compute(OpKernelContext *context) override {
    // get the input data, i.e. encoded mp3 data
    const Tensor &input_tensor = context->input(0);
    const string &input_data = input_tensor.scalar<tstring>()();

    // initialize mp3 decoder
    static mp3dec_t mp3dec;
    mp3dec_init(&mp3dec);

    // decode mp3
    static mp3dec_file_info_t mp3;
    mp3dec_load_buf(&mp3dec, (const uint8_t *)input_data.data(),
                    input_data.size(), &mp3, NULL /* progress callback */,
                    NULL /* user data */);

    // some bookkeeping for generating the output
    desired_channels = desired_channels < 0 ? mp3.channels : desired_channels;
    int single_channel_samples = mp3.samples / mp3.channels;
    desired_samples = desired_samples < 0 ? single_channel_samples : desired_samples;

    // create the outputs
    // samples
    Tensor *output_tensor = nullptr;
    TensorShape output_shape {desired_channels, desired_samples};
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &output_tensor));
    auto output_flat = output_tensor->flat<float>();

    // sample rate
    Tensor *sample_rate_output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({}),
                                                     &sample_rate_output));
    sample_rate_output->flat<int32>()(0) = mp3.hz;

    // copy data into output tensor
    for (int channel = 0; channel < desired_channels; channel++) {
      float *target = output_flat.data() + desired_samples * channel;
      float *source = mp3.buffer;
      if (channel == 1 && mp3.channels == 2)
        // we want 2 channels and we have 2 channels
        source += single_channel_samples;

      int to_copy = std::min(single_channel_samples, desired_samples);
      std::memcpy(target, source, to_copy);

      // fill rest with 0s if necessary
      int to_fill = desired_samples - to_copy;
      if (to_fill > 0) {
        std::memset(target + to_copy, 0, to_fill);
      }
    }
  }

private:
  int32 desired_channels;
  int32 desired_samples;
};

REGISTER_KERNEL_BUILDER(Name("DecodeMp3").Device(DEVICE_CPU), DecodeMp3Op);
