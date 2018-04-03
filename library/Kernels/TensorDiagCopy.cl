kernel void tensorDiagCopy(Tensor input)
{
  int pos = get_global_id(0);
  int y = pos / input_dims[1];
  int x = pos % input_dims[1];
  if (y > x)
      input_data[pos + input_offset] = input_data[x * input_dims[1] + y];
}
