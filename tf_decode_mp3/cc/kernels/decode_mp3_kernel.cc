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
    mp3dec_t mp3dec;
    mp3dec_init(&mp3dec);

    // decode mp3
    mp3dec_file_info_t mp3;
    mp3dec_load_buf(&mp3dec, (const uint8_t *)input_data.data(),
                    input_data.size(), &mp3, NULL /* progress callback */,
                    NULL /* user data */);

    // TODO: add better error handling
    // if MP3 can not be parsed, mp3.channels will be 0 and cause a floating point exception in downstream divisions

    // some bookkeeping for generating the output
    int target_channels = desired_channels < 0 ? mp3.channels : desired_channels;
    int single_channel_samples = mp3.samples / mp3.channels;
    int target_samples = desired_samples < 0 ? single_channel_samples : desired_samples;

    // create the outputs
    // samples
    Tensor *output_tensor = nullptr;
    TensorShape output_shape {target_channels, target_samples};
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, output_shape, &output_tensor));
    auto output_flat = output_tensor->flat<float>();

    // sample rate
    Tensor *sample_rate_output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(1, TensorShape({}),
                                                     &sample_rate_output));
    sample_rate_output->flat<int32>()(0) = mp3.hz;

    // copy data into output tensor
    for (int channel = 0; channel < target_channels; channel++) {
      float *target = output_flat.data() + target_samples * channel;
      // offset the source to the right channel
      float *source = mp3.buffer + (channel % mp3.channels) * single_channel_samples;

      int to_copy = std::min(single_channel_samples, target_samples);
      std::memcpy(target, source, to_copy);

      // fill rest with 0s if necessary
      int to_fill = target_samples - to_copy;
      if (to_fill > 0) {
        std::memset(target + to_copy, 0, to_fill);
      }
    }

    // clean up
    free((void *) mp3.buffer);
  }

private:
  int32 desired_channels;
  int32 desired_samples;
};

REGISTER_KERNEL_BUILDER(Name("DecodeMp3").Device(DEVICE_CPU), DecodeMp3Op);
