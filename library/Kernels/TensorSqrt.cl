kernel void tensorSqrt(Tensor dest, Tensor src)
{
    int pos = get_global_id(0);
    dest_data[pos + dest_offset] = sqrt(src_data[pos + src_offset]);
}
