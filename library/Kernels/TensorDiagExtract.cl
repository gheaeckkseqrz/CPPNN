kernel void tensorDiagExtract(Tensor input, Tensor output)
{
  int pos = get_global_id(0);
  int y = pos / input_dims[1];
  int x = pos % input_dims[1];
  if (x >= y)
    {
      int i = (y * (2 * input_dims[1] - y + 1)) / 2 + (x - y - 1) + 1;
      output_data[i + output_offset] = input_data[pos + input_offset];
    }

}
