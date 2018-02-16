kernel void convolve(Tensor input, Tensor output, Tensor filter, Tensor bias,
		     int padW, int padH, float dW, float dH, int dilationW, int dilationH)
{
    int pos = get_global_id(0);
    if (input_nbDims == 3 && output_nbDims == 3) // Non batch mode
    {
        int channelSize = output_dims[1] * output_dims[2];
        int c = pos / channelSize;
        int y = (pos % channelSize) / output_dims[2];
        int x = (pos % channelSize) % output_dims[2];

        float sum = 0;
        for (int d = 0 ; d < filter_dims[1] ; ++d)
        {
            for (int i = 0 ; i  < filter_dims[2] ; ++i)
            {
               for (int j = 0 ; j  < filter_dims[3] ; ++j)
               {
                   int filterIdx[4];
                    filterIdx[0] = c; // Output channel
                    filterIdx[1] = d; // Input channel
                    filterIdx[2] = j;
                    filterIdx[3] = i;
                    float kernelWeight = sampleTensor(filter_data, filter_dims, filterIdx, 4, filter_offset);
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
        output_data[pos + output_offset] = sum;
    }
    else //if (input_nb_dim == 4 && output_nb_dim == 4) // Batch mode
    {
        if (pos == 0) printf("Batch mode convolution not implemented\n");
        // Will probably never get implemented
    }
}
