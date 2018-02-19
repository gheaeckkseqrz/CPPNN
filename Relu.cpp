#include <iostream>

#include "OpenCLFuncs.h"
#include "Relu.h"
#include "Tensor.h"

namespace NN
{
  Relu::Relu()
  {
  }

  std::shared_ptr<Input> Relu::forward(std::shared_ptr<Input> const &input)
  {
    std::shared_ptr<Tensor> inputTensor = std::dynamic_pointer_cast<Tensor>(input);
    if (_output == nullptr || std::dynamic_pointer_cast<Tensor>(_output)->getSizes() != inputTensor->getSizes())
    	_output.reset(new Tensor(inputTensor->getSizes()));
    std::shared_ptr<Tensor> outputTensor = std::dynamic_pointer_cast<Tensor>(_output);
    OpenCLFuncs::getInstance()->relu(*inputTensor, *outputTensor, outputTensor->getNbElements());
    return _output;
  }
}
