kernel void tensorTranspose(Tensor input, Tensor output)
{
    int pos = get_global_id(0);
    if (input_nbDims == 2 && output_nbDims == 2)
    {
      int dim1 = pos / output_dims[1];
      int dim2 = pos % output_dims[1];

      int indexes[2];
      indexes[0] = dim2;
      indexes[1] = dim1;
      output_data[pos + output_offset] = sampleTensor(input_data, input_dims, indexes, 2, input_offset);
    }
    else
    {
     	if (pos == 0) printf("Transpose only works on 2 dimentional tensors\n");
    }
}
