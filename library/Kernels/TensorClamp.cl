kernel void tensorClamp(Tensor t, float min, float max)
{
    int pos = get_global_id(0);
    t_data[pos + t_offset] = fmax(fmin(t_data[pos + t_offset], max), min);
}
