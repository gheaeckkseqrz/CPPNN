kernel void tensorMeans(Tensor input, Tensor output)
{
  int pos = get_global_id(0);
  if (input_nbDims == 2)
    {
      int indexes[2];
      indexes[0] = pos;
      float accumulator = 0.0f;

      for (int i=0 ; i < input_dims[1] ; ++i)
        {
          indexes[1] = i;
          accumulator += sampleTensor(input_data, input_dims, indexes, input_nbDims, input_offset);
        }
      output_data[pos + output_offset] = accumulator / input_dims[1];
    }
  else
    {
      if (pos == 0) printf("computeTensorMeans kernel is expecting 2 dimentional tensor\n");
    }
}
