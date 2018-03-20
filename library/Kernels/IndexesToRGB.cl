kernel void indexesToRGB(Tensor input, Tensor colors, Tensor output)
{
  int pos = get_global_id(0);
  int channelSize = output_dims[1] * output_dims[2];
  int currentChannel = pos / channelSize;
  int y = (pos % channelSize) / output_dims[2];
  int x = (pos % channelSize) % output_dims[2];

  output_data[pos + output_offset] = 0;
  int indexesCoord[2];
  indexesCoord[0] = y;
  indexesCoord[1] = x;
  int index = sampleTensor(input_data, input_dims, indexesCoord, input_nbDims, input_offset);

  int colorIndexes[2];
  if (index < 0)
    output_data[pos + output_offset] = 0;
  else
    {
      colorIndexes[0] = index;
      colorIndexes[1] = currentChannel;
      output_data[pos + output_offset] = sampleTensor(colors_data, colors_dims, colorIndexes, colors_nbDims, colors_offset);
    }
}
