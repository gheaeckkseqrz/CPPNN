kernel void tensorElementWiseSub(Tensor input, Tensor output, Tensor o)
{
  int pos = get_global_id(0);
  output_data[pos + output_offset] = input_data[pos + input_offset] - o_data[pos + o_offset];
}
