kernel void tensorDataEquals(Tensor a, Tensor b, Tensor output, float tolerance)
{
  int pos = get_global_id(0);
  if (fabs(a_data[pos + a_offset] - b_data[pos + b_offset]) > tolerance)
    output_data[0] += 1;
}
