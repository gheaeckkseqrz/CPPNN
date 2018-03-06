kernel void convolve(Tensor input, Tensor output, Tensor filter, Tensor bias,
		     int padW, int padH, float dW, float dH, int dilationW, int dilationH,
		     __local float *filterCache, int workgroupSize)
{
    int globalPos = get_global_id(0);
    int localPos = get_local_id(0);
    if (input_nbDims == 3 && output_nbDims == 3) // Non batch mode
    {
      int channelSize = output_dims[1] * output_dims[2];
      int workgroupId = globalPos / workgroupSize;
      int nbWorkgroupPerChannel = (channelSize / workgroupSize) + ((channelSize % workgroupSize) == 0 ? 0 : 1);

      int channelToProcess = workgroupId / nbWorkgroupPerChannel;
      int channelOffset = (workgroupId % nbWorkgroupPerChannel) * workgroupSize;
      int channelPos = channelOffset + localPos;

      int c = channelToProcess;
      int y = channelPos / output_dims[2];
      int x = channelPos % output_dims[2];

      /* CACHING THE CONV FILTER IN LOCAL MEMORY */
      int filterIdx[4];
      filterIdx[0] = c; // Output channel
      filterIdx[1] = 0; // Input channel
      filterIdx[2] = 0;
      filterIdx[3] = 0;
      int filterIndex = tensorIndex(filter_dims, filterIdx, 4, filter_offset);
      async_work_group_copy(filterCache, &filter_data[filterIndex], filter_dims[1] * filter_dims[2] * filter_dims[3], 0);
      barrier(CLK_LOCAL_MEM_FENCE);

      if (x < output_dims[2] && y < output_dims[1])
	{
	  int outputCoordinates[3];
	  outputCoordinates[0] = c;
	  outputCoordinates[1] = y;
	  outputCoordinates[2] = x;
	  int outputIndex = tensorIndex(output_dims, outputCoordinates, 3, output_offset);

	  float sum = 0;
	  for (int d = 0 ; d < filter_dims[1] ; ++d)
	    {
	      for (int i = 0 ; i  < filter_dims[2] ; ++i)
	  	{
	  	  for (int j = 0 ; j  < filter_dims[3] ; ++j)
	  	    {
	  	      int filterIdx[3];
	  	      filterIdx[0] = d; // Input channel
	  	      filterIdx[1] = j;
	  	      filterIdx[2] = i;
	  	      int kernelIndex = tensorIndex(&filter_dims[1], filterIdx, 3, 0);
	  	      float kernelWeight = filterCache[kernelIndex];
	  	      float inputValue = sample3DTensorWithPadding(input_data, input_dims, d,
	  	      						   (float)x * dW + (i * dilationW) - padW,
	  	      						   (float)y * dH + (j * dilationH) - padH,
	  	      						   input_offset);
	  	      sum += (kernelWeight * inputValue);
	  	    }
	  	}
	    }
	  int biasIdx[1];
	  biasIdx[0] = c;
	  float b = sampleTensor(bias_data, bias_dims, biasIdx, 1, bias_offset);
	  sum += b;
	  output_data[outputIndex] = sum;
	}
    }
    else //if (input_nb_dim == 4 && output_nb_dim == 4) // Batch mode
      {
        if (globalPos == 0) printf("Batch mode convolution not implemented\n");
        // Will probably never get implemented
      }
}
