kernel void tensorFill(Tensor t, float val)
{
    int pos = get_global_id(0);
    t_data[pos + t_offset] = val;
}
