kernel void tensorPow(Tensor input, Tensor output, float v)
{
  int pos = get_global_id(0);
  output_data[pos + output_offset] = pow(input_data[pos + input_offset], v);
}
