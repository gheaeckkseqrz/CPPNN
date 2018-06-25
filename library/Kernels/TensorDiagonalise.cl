kernel void tensorDiagonalise(Tensor input, Tensor output)
{
  int pos = get_global_id(0);
  int y = pos / output_dims[1];
  int x = pos % output_dims[1];
  if (y == x)
    output_data[pos + output_offset] = input_data[x + input_offset];
  else
    output_data[pos + output_offset] = 0;
}
