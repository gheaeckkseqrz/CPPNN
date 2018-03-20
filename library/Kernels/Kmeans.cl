kernel void kmeans(Tensor input, Tensor centroids, Tensor output, int mode)
{
  // 0 : return index of closest centroid
  // 1 : return distance to closest centroid
  // 2 : return sum of distances to all centroids
  int pos = get_global_id(0);
  int bestIndex = -1;
  float best = -1;
  float loss = 0;
  for (int c = 0 ; c < centroids_dims[0] ; ++c) // Loop centroids
    {
      if (mode != 2)
        loss = 0;
      for (int i = 0 ; i < input_dims[0] ; ++i) // Loss
        {
          int inputIndexes[2];
          inputIndexes[0] = i;
          inputIndexes[1] = pos;
          float valData = sampleTensor(input_data, input_dims, inputIndexes, input_nbDims, input_offset);
          int centroidsIndexes[2];
          centroidsIndexes[0] = c;
          centroidsIndexes[1] = i;
          float centroidData = sampleTensor(centroids_data, centroids_dims, centroidsIndexes, centroids_nbDims, centroids_offset);
          loss += pow(centroidData - valData, 2); // L2
        }
      if (loss < best || bestIndex < 0)
        {
          best = loss;
          bestIndex = c;
        }
    }
  if (mode == 0)
    output_data[pos + output_offset] = (float)bestIndex;
  else if (mode == 1)
    output_data[pos + output_offset] = best;
  else
    output_data[pos + output_offset] = loss;
}
