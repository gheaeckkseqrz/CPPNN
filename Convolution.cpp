#include <iostream>
#include <cmath>

#include "Convolution.h"
#include "OpenCLFuncs.h"

namespace NN
{
  Convolution::Convolution()
  {
  }

  Convolution::~Convolution()
  {
  }

  void Convolution::setFilter(std::shared_ptr<Tensor> const &filter, std::shared_ptr<Tensor> const &bias)
  {
    _filter = filter;
    if (bias == nullptr)
      {
	std::vector<float> defaultBias(_filter->getSizes()[0], 0);
	_bias = std::make_shared<Tensor>(defaultBias);
      }
    else
      _bias = bias;
  }

  std::shared_ptr<Input> Convolution::forward(std::shared_ptr<Input> const &input)
  {
    Tensor *inputTensor = dynamic_cast<Tensor*>(input.get());
    assert(inputTensor->getSizes().size() == 3);
    std::vector<int> outputSizes(3, 0);
    outputSizes[0] = _filter->getSize(0);
    outputSizes[1] = (int)ceil((double)(inputTensor->getSize(1) - ((int)(_filter->getSize(2) / 2) * 2) + (_padH * 2)) / _dW);
    outputSizes[2] = (int)ceil((double)(inputTensor->getSize(2) - ((int)(_filter->getSize(3) / 2) * 2) + (_padW * 2)) / _dH);

    if (_output == nullptr || dynamic_cast<Tensor*>(_output.get())->getSizes() != outputSizes)
      {
	_output.reset(new Tensor(outputSizes));
      }
    Tensor *outputTensor = dynamic_cast<Tensor*>(_output.get());
    OpenCLFuncs::getInstance()->convolve(*inputTensor, *outputTensor, *_filter.get(), *_bias.get(),
    					 _padW, _padH, _dW, _dH, 0, 0, outputTensor->getNbElements());
    return _output;
  }
}
