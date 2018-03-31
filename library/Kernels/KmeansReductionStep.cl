kernel void kmeansReductionStep(Tensor input, int offset, int length)
{
  int pos = get_global_id(0);
  int total = 1;
  for (int i = 0 ; i < input_nbDims ; ++i)
    total *= input_dims[i];
  int channelSize = total / input_dims[0];
  int channel = pos / channelSize;
  int point = pos % channelSize;

  if (point < offset && point + offset < length)
      input_data[pos + input_offset] += input_data[pos + input_offset + offset];
}
