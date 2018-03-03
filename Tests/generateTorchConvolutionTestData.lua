require 'nn'

-- CONVOLUTION 1 IN 1 OUT NO PADDING NO BIAS
local convolution1in1outNoPaddingNoBias = nn.SpatialConvolution(1, 1, 3, 3, 1, 1, 0, 0):noBias():float()
torch.save('convolution1in1outNoPaddingNoBias.t7', convolution1in1outNoPaddingNoBias, 'ascii')
local convolution1in1outNoPaddingNoBiasResult = convolution1in1outNoPaddingNoBias:forward(input1Channel)
torch.save('convolution1in1outNoPaddingNoBiasResult.t7', convolution1in1outNoPaddingNoBiasResult, 'ascii')

-- CONVOLUTION 1 IN 1 OUT PADDING NO BIAS
local convolution1in1outPaddingNoBias = nn.SpatialConvolution(1, 1, 3, 3, 1, 1, 1, 1):noBias():float()
torch.save('convolution1in1outPaddingNoBias.t7', convolution1in1outPaddingNoBias, 'ascii')
local convolution1in1outPaddingNoBiasResult = convolution1in1outPaddingNoBias:forward(input1Channel)
torch.save('convolution1in1outPaddingNoBiasResult.t7', convolution1in1outPaddingNoBiasResult, 'ascii')

-- CONVOLUTION 1 IN 1 OUT NO PADDING BIAS
local convolution1in1outNoPaddingBias = nn.SpatialConvolution(1, 1, 3, 3, 1, 1, 0, 0):float()
torch.save('convolution1in1outNoPaddingBias.t7', convolution1in1outNoPaddingBias, 'ascii')
local convolution1in1outNoPaddingBiasResult = convolution1in1outNoPaddingBias:forward(input1Channel)
torch.save('convolution1in1outNoPaddingBiasResult.t7', convolution1in1outNoPaddingBiasResult, 'ascii')

-- CONVOLUTION 1 IN 1 OUT PADDING BIAS
local convolution1in1outPaddingBias = nn.SpatialConvolution(1, 1, 3, 3, 1, 1, 1, 1):float()
torch.save('convolution1in1outPaddingBias.t7', convolution1in1outPaddingBias, 'ascii')
local convolution1in1outPaddingBiasResult = convolution1in1outPaddingBias:forward(input1Channel)
torch.save('convolution1in1outPaddingBiasResult.t7', convolution1in1outPaddingBiasResult, 'ascii')

-- CONVOLUTION IN < OUT PADDING NO BIAS
local convolutionInLTOutPaddingNoBias = nn.SpatialConvolution(1, 3, 3, 3, 1, 1, 1, 1):float()
torch.save('convolutionInLTOutPaddingNoBias.t7', convolutionInLTOutPaddingNoBias, 'ascii')
local convolutionInLTOutPaddingNoBiasResult = convolutionInLTOutPaddingNoBias:forward(input1Channel)
torch.save('convolutionInLTOutPaddingNoBiasResult.t7', convolutionInLTOutPaddingNoBiasResult, 'ascii')

-- CONVOLUTION IN < OUT PADDING NO BIAS
local convolutionInLTOutNoPaddingNoBias = nn.SpatialConvolution(1, 3, 3, 3, 1, 1, 0, 0):float()
torch.save('convolutionInLTOutNoPaddingNoBias.t7', convolutionInLTOutNoPaddingNoBias, 'ascii')
local convolutionInLTOutNoPaddingNoBiasResult = convolutionInLTOutNoPaddingNoBias:forward(input1Channel)
torch.save('convolutionInLTOutNoPaddingNoBiasResult.t7', convolutionInLTOutNoPaddingNoBiasResult, 'ascii')

-- CONVOLUTION IN > OUT PADDING NO BIAS
local convolutionInGTOutPaddingNoBias = nn.SpatialConvolution(3, 1, 3, 3, 1, 1, 1, 1):float()
torch.save('convolutionInGTOutPaddingNoBias.t7', convolutionInGTOutPaddingNoBias, 'ascii')
local convolutionInGTOutPaddingNoBiasResult = convolutionInGTOutPaddingNoBias:forward(input)
torch.save('convolutionInGTOutPaddingNoBiasResult.t7', convolutionInGTOutPaddingNoBiasResult, 'ascii')

-- CONVOLUTION IN > OUT PADDING NO BIAS
local convolutionInGTOutNoPaddingNoBias = nn.SpatialConvolution(3, 1, 3, 3, 1, 1, 0, 0):float()
torch.save('convolutionInGTOutNoPaddingNoBias.t7', convolutionInGTOutNoPaddingNoBias, 'ascii')
local convolutionInGTOutNoPaddingNoBiasResult = convolutionInGTOutNoPaddingNoBias:forward(input)
torch.save('convolutionInGTOutNoPaddingNoBiasResult.t7', convolutionInGTOutNoPaddingNoBiasResult, 'ascii')

-- CONVOLUTION IN < OUT PADDING
local convolutionInLTOutPaddingBias = nn.SpatialConvolution(1, 3, 3, 3, 1, 1, 1, 1):float()
torch.save('convolutionInLTOutPaddingBias.t7', convolutionInLTOutPaddingBias, 'ascii')
local convolutionInLTOutPaddingBiasResult = convolutionInLTOutPaddingBias:forward(input1Channel)
torch.save('convolutionInLTOutPaddingBiasResult.t7', convolutionInLTOutPaddingBiasResult, 'ascii')

-- CONVOLUTION IN < OUT PADDING
local convolutionInLTOutNoPaddingBias = nn.SpatialConvolution(1, 3, 3, 3, 1, 1, 0, 0):float()
torch.save('convolutionInLTOutNoPaddingBias.t7', convolutionInLTOutNoPaddingBias, 'ascii')
local convolutionInLTOutNoPaddingBiasResult = convolutionInLTOutNoPaddingBias:forward(input1Channel)
torch.save('convolutionInLTOutNoPaddingBiasResult.t7', convolutionInLTOutNoPaddingBiasResult, 'ascii')

-- CONVOLUTION IN > OUT PADDING
local convolutionInGTOutPaddingBias = nn.SpatialConvolution(3, 1, 3, 3, 1, 1, 1, 1):float()
torch.save('convolutionInGTOutPaddingBias.t7', convolutionInGTOutPaddingBias, 'ascii')
local convolutionInGTOutPaddingBiasResult = convolutionInGTOutPaddingBias:forward(input)
torch.save('convolutionInGTOutPaddingBiasResult.t7', convolutionInGTOutPaddingBiasResult, 'ascii')

-- CONVOLUTION IN > OUT PADDING
local convolutionInGTOutNoPaddingBias = nn.SpatialConvolution(3, 1, 3, 3, 1, 1, 0, 0):float()
torch.save('convolutionInGTOutNoPaddingBias.t7', convolutionInGTOutNoPaddingBias, 'ascii')
local convolutionInGTOutNoPaddingBiasResult = convolutionInGTOutNoPaddingBias:forward(input)
torch.save('convolutionInGTOutNoPaddingBiasResult.t7', convolutionInGTOutNoPaddingBiasResult, 'ascii')
