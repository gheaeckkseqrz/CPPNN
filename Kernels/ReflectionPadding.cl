kernel void reflectionPadding(Tensor input, Tensor output, int padL, int padR, int padT, int padB)
{
    int pos = get_global_id(0);

    if (input_nbDims== 3) // Non-Batch Mode
    {
        int channelSize = output_dims[1] * output_dims[2];
        int c = pos / channelSize;
        int y = (pos % channelSize) / output_dims[2];
        int x = (pos % channelSize) % output_dims[2];
        int srcX = abs(x - padL);
        int srcY = abs(y - padT);
        if (srcY >= input_dims[1])
            srcY -= 2 * (srcY - input_dims[1] + 1);
        if (srcX >= input_dims[2])
            srcX -= 2 * (srcX - input_dims[2] + 1);
        float v = sample3DTensor(input_data, input_dims, c, srcX, srcY, input_offset);
        output_data[pos + output_offset] = v;
    }
    else // Batch Mode
    {
      // Will probably never get implemented
    }
}
