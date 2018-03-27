#include <iostream>

#include "OpenCLFuncs.h"
#include "Relu.h"
#include "Tensor.h"

namespace NN
{
  Relu::Relu()
  {
  }

  std::shared_ptr<Tensor> Relu::forward(std::shared_ptr<Tensor> const &input)
  {
    if (_output == nullptr || _output->getSizes() != input->getSizes())
    	_output.reset(new Tensor(input->getSizes()));
    OpenCLFuncs::getInstance()->relu(*input, *_output, _output->getNbElements());
    return _output;
  }

  std::string Relu::print() const
  {
    return "Relu";
  }
}
