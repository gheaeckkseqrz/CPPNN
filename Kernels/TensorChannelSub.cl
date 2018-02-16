kernel void tensorChannelSub(Tensor input, Tensor output)
{
  int pos = get_global_id(0);
  int channelSize = 1;
  for (int i = 1 ; i < output_nbDims ; ++i)
    channelSize *= output_dims[i];
  int currentChannel = pos / channelSize;
  output_data[pos + output_offset] -= input_data[currentChannel + input_offset];
}
