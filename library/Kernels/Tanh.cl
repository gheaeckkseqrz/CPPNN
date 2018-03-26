kernel void hard_tan(Tensor input, Tensor output)
{
  int pos = get_global_id(0);
  output_data[pos + output_offset] = tanh(input_data[pos + input_offset]);
}
