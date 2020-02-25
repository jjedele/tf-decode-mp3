#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;
using tensorflow::shape_inference::DimensionHandle;
using tensorflow::shape_inference::InferenceContext;
using tensorflow::shape_inference::ShapeHandle;

Status DecodeMp3ShapeFn(InferenceContext *c) {
  ShapeHandle unused;
  TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 0, &unused));

  DimensionHandle channels_dim;
  int32 desired_channels;
  TF_RETURN_IF_ERROR(c->GetAttr("desired_channels", &desired_channels));
  if (desired_channels == -1) {
    channels_dim = c->UnknownDim();
  } else {
    if (desired_channels < 0) {
      return errors::InvalidArgument("channels must be non-negative, got ",
                                     desired_channels);
    }
    channels_dim = c->MakeDim(desired_channels);
  }
  DimensionHandle samples_dim;
  int32 desired_samples;
  TF_RETURN_IF_ERROR(c->GetAttr("desired_samples", &desired_samples));
  if (desired_samples == -1) {
    samples_dim = c->UnknownDim();
  } else {
    if (desired_samples < 0) {
      return errors::InvalidArgument("samples must be non-negative, got ",
                                     desired_samples);
    }
    samples_dim = c->MakeDim(desired_samples);
  }
  c->set_output(0, c->MakeShape({samples_dim, channels_dim}));
  c->set_output(1, c->Scalar());
  return Status::OK();
}

REGISTER_OP("DecodeMp3")
    .Input("encoded: string")
    .Attr("desired_channels: int = -1")
    .Attr("desired_samples: int = -1")
    .Output("samples: float32")
    .Output("sample_rate: int32")
    .SetShapeFn(DecodeMp3ShapeFn);
