kernel void tensorElementWiseDiv(Tensor input, Tensor output)
{
  int pos = get_global_id(0);
  output_data[pos + output_offset] /= input_data[pos + input_offset];
}
