kernel void pixelShuffle(Tensor input, Tensor output, int upscaleFactor)
{
  int pos = get_global_id(0);
  int upscaleFactorSquarred = upscaleFactor * upscaleFactor;

  if (input_nbDims == 3 && output_nbDims == 3)
    {
      int channelSize = output_dims[1] * output_dims[2];
      int c = pos / channelSize;
      int y = (pos % channelSize) / output_dims[2];
      int x = (pos % channelSize) % output_dims[2];

      int squareIdx = (y % upscaleFactor) * upscaleFactor + (x % upscaleFactor);
      int ichannelSize = input_dims[1] * input_dims[2];
      int sampleC = squareIdx * (input_dims[0] / upscaleFactorSquarred) + c;
      int sampleX = x / upscaleFactor;
      int sampleY = y / upscaleFactor;

      output_data[pos + output_offset] = sample3DTensor(input_data, input_dims, sampleC, sampleX, sampleY, input_offset);
    }
}
