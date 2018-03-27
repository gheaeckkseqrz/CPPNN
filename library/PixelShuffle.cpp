#include <cmath>
#include "OpenCLFuncs.h"
#include "PixelShuffle.h"
#include "Tensor.h"

namespace NN
{
  PixelShuffle::PixelShuffle()
  {
    _upscaleFactor = 2;
    _upscaleFactorSquarred = pow(_upscaleFactor, 2);
  }

  std::shared_ptr<Tensor> PixelShuffle::forward(std::shared_ptr<Tensor> const &input)
  {
    std::vector<int> inputSizes = input->getSizes();
    std::vector<int> outputSizes(3);
    outputSizes[0] = input->getSize(0) / _upscaleFactorSquarred;
    outputSizes[1] = input->getSize(1) * _upscaleFactor;
    outputSizes[2] = input->getSize(2) * _upscaleFactor;

    if (_output == nullptr || _output->getSizes() != outputSizes)
      _output = std::make_shared<Tensor>(outputSizes);

    OpenCLFuncs::getInstance()->pixelShuffle(*input, *_output, _upscaleFactor, _output->getNbElements());
    return _output;
  }

  std::string PixelShuffle::print() const
  {
    return "PixelShuffle";
  }
}
