#include <cmath>
#include <iostream>
#include <stdexcept>

#include "Convolution.h"
#include "OpenCLFuncs.h"

namespace NN
{
  Convolution::Convolution()
  {
    _padW = 1;
    _padH = 1;
    _dW = 1;
    _dH = 1;
    _dilationW = 1;
    _dilationH = 1;
  }

  Convolution::~Convolution()
  {
  }

  void Convolution::setFilter(std::shared_ptr<Tensor> const &filter, std::shared_ptr<Tensor> const &bias)
  {
    _filter = std::make_shared<Tensor>(filter->getSizes(), filter->read(), CL_MEM_READ_ONLY);
    if (bias == nullptr)
      {
	std::vector<float> defaultBias(_filter->getSizes()[0], 0);
	_bias = std::make_shared<Tensor>(defaultBias);
      }
    else
      _bias = bias;
  }

  void Convolution::setPadding(int padW, int padH)
  {
    _padW = padW;
    _padH = padH;
  }

  std::shared_ptr<Input> Convolution::forward(std::shared_ptr<Input> const &input)
  {
    std::shared_ptr<Tensor> inputTensor = std::dynamic_pointer_cast<Tensor>(input);
    if (_filter == nullptr || _bias == nullptr)
      throw std::runtime_error("Convolution isn't setup");
    if (inputTensor == nullptr || inputTensor->getSizes().size() != 3)
      throw std::runtime_error("Convolution recieved invalid input");
    if (_filter->getSize(1) != inputTensor->getSize(0))
	throw std::runtime_error("Convolution recieved invalid input for filter");
    std::vector<int> outputSizes(3, 0);
    outputSizes[0] = _filter->getSize(0);
    outputSizes[1] = (int)ceil((double)(inputTensor->getSize(1) - ((int)(_filter->getSize(2) / 2) * 2) + (_padH * 2)) / _dW);
    outputSizes[2] = (int)ceil((double)(inputTensor->getSize(2) - ((int)(_filter->getSize(3) / 2) * 2) + (_padW * 2)) / _dH);

    if (_output == nullptr || std::dynamic_pointer_cast<Tensor>(_output)->getSizes() != outputSizes)
	_output.reset(new Tensor(outputSizes));
    std::shared_ptr<Tensor> outputTensor = std::dynamic_pointer_cast<Tensor>(_output);
    outputTensor->fill(42);

    unsigned int workGroupSize = 1024;
    unsigned int outputChannelSize = outputTensor->getNbElements() / outputTensor->getSize(0);
    unsigned int nbWorkgroupPerChannel = (outputChannelSize / workGroupSize) + ((outputChannelSize % workGroupSize) == 0 ? 0 : 1);
    unsigned int totalNbOfWorkItems = outputTensor->getSize(0) * nbWorkgroupPerChannel * workGroupSize;
    OpenCLFuncs::getInstance()->convolve(*inputTensor, *outputTensor, *_filter, *_bias,
					 _padW, _padH, _dW, _dH, _dilationW, _dilationH,
					 workGroupSize,
					 totalNbOfWorkItems, workGroupSize);
    return _output;
  }
}
