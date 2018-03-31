kernel void tensorToMat(Tensor input, Tensor output)
{
  int pos = get_global_id(0);
  int channelSize = input_dims[1] * input_dims[2];
  int c = pos % input_dims[0];
  int y = (pos / input_dims[0]) / input_dims[1];
  int x = (pos / input_dims[0]) % input_dims[1];

  int BRG2RGB[3];
  BRG2RGB[0] = 2;
  BRG2RGB[1] = 1;
  BRG2RGB[2] = 0;

  int src = (BRG2RGB[c] * channelSize) + (y * input_dims[2]) + x;
  output_data[pos + output_offset] = input_data[src + input_offset];
}
