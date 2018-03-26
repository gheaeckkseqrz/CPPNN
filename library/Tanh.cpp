#include "OpenCLFuncs.h"
#include "Tanh.h"
#include "Tensor.h"

namespace NN
{
  Tanh::Tanh()
  {
  }

  std::shared_ptr<Input> Tanh::forward(std::shared_ptr<Input> const &input)
  {
    std::shared_ptr<Tensor> inputTensor = std::dynamic_pointer_cast<Tensor>(input);
    if (_output == nullptr || std::dynamic_pointer_cast<Tensor>(_output)->getSizes() != inputTensor->getSizes())
      _output.reset(new Tensor(inputTensor->getSizes()));
    std::shared_ptr<Tensor> outputTensor = std::dynamic_pointer_cast<Tensor>(_output);
    OpenCLFuncs::getInstance()->hard_tan(*inputTensor, *outputTensor, outputTensor->getNbElements());
    return _output;
  }

  std::string Tanh::print() const
  {
    return "Tanh";
  }
}
