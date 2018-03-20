kernel void tensorCopy(Tensor dest, Tensor src)
{
    int pos = get_global_id(0);
    dest_data[pos + dest_offset] = src_data[pos + src_offset];
}
