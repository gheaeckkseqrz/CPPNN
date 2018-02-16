kernel void tensorValueMul(Tensor input, Tensor output, float val)
{
    int pos = get_global_id(0);
    output_data[pos + output_offset] = input_data[pos + input_offset] * val;
}
