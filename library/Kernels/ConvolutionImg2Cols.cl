kernel void convolutionImg2Cols(Tensor input, Tensor output, int filterW, int filterH, int padW, int padH, int dilationW, int dilationH)
{
  int pos = get_global_id(0);
  if (input_nbDims == 3 && output_nbDims == 2) // Non batch mode
    {
      int y = pos / output_dims[1];
      int x = pos % output_dims[1]; // -- Aka wich point in the input image
      int effectiveInputW = input_dims[2] - (2 * (filterW / 2 * dilationH)) + (2 * padW);
      int inY = (x / effectiveInputW) + ((filterH / 2 * dilationH) - padH);
      int inX = (x % effectiveInputW) + ((filterW / 2 * dilationW) - padW);;
      int inC = y / (filterW * filterH);
      int inDY = ((y % (filterW * filterH) / filterW) - (filterH / 2)) * dilationH;
      int inDX = ((y % (filterW * filterH) % filterW) - (filterW / 2)) * dilationW;
      output_data[pos] = sample3DTensorWithPadding(input_data, input_dims, inC, inX + inDX, inY + inDY, input_offset);
    }
  else
    {
      if (pos == 0)
	{
	  printf("ConvolutionImg2Cols Kernel not receiving the right arguments\n");
	}
    }
}
