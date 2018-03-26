kernel void relu(Tensor input, Tensor output)
{
  int pos = get_global_id(0);
  output_data[pos + output_offset] = fmax(input_data[pos + input_offset], 0.0f);
}
