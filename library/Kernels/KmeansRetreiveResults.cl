kernel void kmeansRetreiveResults(Tensor scratch, Tensor centroids)
{
  int pos = get_global_id(0);
  centroids_data[pos + centroids_offset] = scratch_data[scratch_offset + (pos * scratch_dims[1])] / scratch_data[scratch_offset + (centroids_dims[0] * scratch_dims[1])];
}
