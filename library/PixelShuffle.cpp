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

  std::shared_ptr<Input> PixelShuffle::forward(std::shared_ptr<Input> const &input)
  {
    std::shared_ptr<Tensor> t = std::dynamic_pointer_cast<Tensor>(input);
    std::vector<int> inputSizes = t->getSizes();
    std::vector<int> outputSizes(3);
    outputSizes[0] = t->getSize(0) / _upscaleFactorSquarred;
    outputSizes[1] = t->getSize(1) * _upscaleFactor;
    outputSizes[2] = t->getSize(2) * _upscaleFactor;

    if (_output == nullptr || std::dynamic_pointer_cast<Tensor>(_output)->getSizes() != outputSizes)
      _output = std::make_shared<Tensor>(outputSizes);

    std::shared_ptr<Tensor> outputTensor = std::dynamic_pointer_cast<Tensor>(_output);
    OpenCLFuncs::getInstance()->pixelShuffle(*t, *outputTensor, _upscaleFactor, outputTensor->getNbElements());
    return _output;
  }

  std::string PixelShuffle::print() const
  {
    return "PixelShuffle";
  }
}
