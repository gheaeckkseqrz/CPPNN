require 'image'

input = image.lena():float()
torch.save('input.t7', input, 'ascii')
input1Channel = input:sum(1):div(3)
torch.save('input1Channel.t7', input1Channel, 'ascii')

require './generateTorchConvolutionTestData'
require './generateTorchVGGTestData'
