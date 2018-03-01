kernel void tensorChannelMul(Tensor input, Tensor output, Tensor channelMul)
{
  int pos = get_global_id(0);
  int channelSize = 1;
  for (int i = 1 ; i < output_nbDims ; ++i)
    channelSize *= output_dims[i];
  int currentChannel = pos / channelSize;
  output_data[pos + output_offset] = input_data[pos + input_offset] * channelMul_data[channelMul_offset + currentChannel];
}
