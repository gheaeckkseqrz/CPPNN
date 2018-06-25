kernel void spatialUpsamplingNearest(Tensor input, Tensor output, int upscaleFactor)
{
  int pos = get_global_id(0);
  if (input_nbDims == 3 && output_nbDims == 3)
    {
      int channelSize = output_dims[1] * output_dims[2];
      int c = pos / channelSize;
      int y = (pos % channelSize) / output_dims[2];
      int x = (pos % channelSize) % output_dims[2];
      output_data[pos + output_offset] = sample3DTensor(input_data, input_dims, c, x / upscaleFactor, y / upscaleFactor, input_offset);
    }
}
