
inline int tensorIndex(__global  int *t_dims, int *indexes, int t_nb_dim, int offset)
{
  int index = 0;
  int stride = 1;
  for (int i=t_nb_dim-1 ; i >= 0 ; --i)
    {
      index += indexes[i] * stride;
      stride *= t_dims[i];
    }
  return index;
}

inline float sampleTensor(__global  float *t, __global  int *t_dims, int *indexes, int t_nb_dim, int offset)
{
  int index = tensorIndex(t_dims, indexes, t_nb_dim, offset);
  return t[index + offset];
}

inline float sampleTensorWithPadding(__global  float *t, __global  int *t_dims, int *indexes, int t_nb_dim, int offset)
{
  for (int i = 0 ; i < t_nb_dim ; ++i)
    if (indexes[i] >= t_dims[i] || indexes[i] < 0)
      return 0.0f;
  return sampleTensor(t, t_dims, indexes, t_nb_dim, offset);
}

inline float sample3DTensor(__global  float *t, __global  int *t_dims, int c, int x, int y, int offset)
{
    int idx[3];
    idx[0] = c;
    idx[1] = y;
    idx[2] = x;
    return sampleTensor(t, t_dims, idx, 3, offset);
}

inline float sample3DTensorWithPadding(__global  float *t, __global  int *t_dims, int c, int x, int y, int offset)
{
    if (x < 0 || y < 0 || x >= t_dims[2] || y >= t_dims[1])
        return 0.0f;
    return sample3DTensor(t, t_dims, c, x, y, offset);
}
