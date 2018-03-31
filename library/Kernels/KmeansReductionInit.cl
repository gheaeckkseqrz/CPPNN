kernel void kmeansReductionInit(Tensor data, Tensor scratch, Tensor indexes, float targetIndex)
{
  int pos = get_global_id(0);
  int total = 1;
  for (int i = 0 ; i < data_nbDims ; ++i)
    total *= data_dims[i];
  int channelSize = total / data_dims[0];
  int channel = pos / channelSize;
  int point = pos % channelSize;

  if (channel < data_dims[0])
    scratch_data[pos + scratch_offset] = data_data[pos + data_offset] * ((indexes_data[point + indexes_offset] == targetIndex) ? 1 : 0);
  else
    scratch_data[pos + scratch_offset] = ((indexes_data[point + indexes_offset] == targetIndex) ? 1 : 0);
}
