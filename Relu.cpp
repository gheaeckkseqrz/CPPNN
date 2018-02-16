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
    Tensor *inputTensor = dynamic_cast<Tensor*>(input.get());
    if (_output == nullptr || dynamic_cast<Tensor*>(_output.get())->getSizes() != inputTensor->getSizes())
    	_output.reset(new Tensor(inputTensor->getSizes()));
    Tensor *outputTensor = dynamic_cast<Tensor*>(_output.get());
    OpenCLFuncs::getInstance()->relu(*inputTensor, *outputTensor, outputTensor->getNbElements());
    return _output;
  }
}
