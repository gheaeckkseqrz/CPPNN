kernel void tensorFromMat(Tensor input, Tensor output)
{
  int pos = get_global_id(0);
  int channelSize = output_dims[1] * output_dims[2];
  int c = pos / channelSize;
  int y = (pos % channelSize) / output_dims[2];
  int x = (pos % channelSize) % output_dims[2];

  int BRG2RGB[3];
  BRG2RGB[0] = 2;
  BRG2RGB[1] = 1;
  BRG2RGB[2] = 0;

  int src = (y * input_dims[1] + x) * input_dims[0] + BRG2RGB[c];
  output_data[pos + output_offset] = input_data[src + input_offset];
}
