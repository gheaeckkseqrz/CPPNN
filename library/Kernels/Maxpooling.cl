kernel void maxPooling(Tensor input, Tensor output)
{
    int pos = get_global_id(0);
    if (input_nbDims == 3 && output_nbDims == 3) // Non batch mode
    {
        int channelSize = output_dims[1] * output_dims[2];
        int c = pos / channelSize;
        int y = (pos % channelSize) / output_dims[2];
        int x = (pos % channelSize) % output_dims[2];

        float maxVal = sample3DTensor(input_data, input_dims, c, x * 2, y * 2, input_offset);
        for (int i = 0 ; i < 2 ; ++i)
        {
            for (int j = 0 ; j < 2 ; ++j)
            {
                maxVal = fmax(sample3DTensor(input_data, input_dims, c, x * 2 + i, y * 2 + j, input_offset), maxVal);
            }
        }
        output_data[pos] = maxVal;
    }
    if (input_nbDims == 4 && output_nbDims == 4) // Batch mode
    {
        // Will probably never get implemented
    }

}
