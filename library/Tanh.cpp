#include "OpenCLFuncs.h"
#include "Tanh.h"
#include "Tensor.h"

namespace NN
{
  Tanh::Tanh()
  {
  }

  std::shared_ptr<Tensor> Tanh::forward(std::shared_ptr<Tensor> const &input)
  {
    if (_output == nullptr || _output->getSizes() != input->getSizes())
      _output = std::make_shared<Tensor>(input->getSizes());
    OpenCLFuncs::getInstance()->hard_tan(*input, *_output, _output->getNbElements());
    return _output;
  }

  std::string Tanh::print() const
  {
    return "Tanh";
  }
}
