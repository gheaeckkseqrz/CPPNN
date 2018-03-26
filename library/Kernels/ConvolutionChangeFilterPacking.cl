kernel void convolutionChangeFilterPacking(Tensor input, Tensor output)
{
  // Input format [kH x kW x NbInputChannels x NbOutputChannels]
  // Output format [NbOutputChannels x NbInputChannels x kH x kW]

  int pos = get_global_id(0);

  int ocs = output_dims[1] * output_dims[2] * output_dims[3]; // Size of outputChannel
  int ics = output_dims[2] * output_dims[3]; // Size of InputChannel
  int h = output_dims[2];
  int w = output_dims[3];
  int oc = pos / ocs; // 0 < 64
  int ic = pos % ocs / ics; // 0 < 3
  int y = pos % ics / w;  // 0 < 3
  int x = pos % w; // 0 < 3

  int indexes [4];
  indexes[0] = y; // H
  indexes[1] = x; // W
  indexes[2] = ic; // IN
  indexes[3] = oc; // OUT
  output_data[pos] = sampleTensor(input_data, input_dims, indexes, input_nbDims, input_offset);
}
