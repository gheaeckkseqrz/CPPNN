#include "OpenCLFuncs.h"
#include "ReflectionPadding.h"
#include "Tensor.h"

namespace NN
{
  ReflectionPadding::ReflectionPadding()
  {
  }

  std::shared_ptr<Tensor> ReflectionPadding::forward(std::shared_ptr<Tensor> const &input)
  {
    assert(input->getSizes().size() == 3);
    std::vector<int> outputSizes(3, 0);
    outputSizes[0] = input->getSize(0);
    outputSizes[1] = input->getSize(1) + _padT + _padB;
    outputSizes[2] = input->getSize(2) + _padL + _padR;

    if (_output == nullptr || _output->getSizes() != outputSizes)
	_output.reset(new Tensor(outputSizes));
    OpenCLFuncs::getInstance()->reflectionPadding(*input, *_output, _padL, _padR, _padT, _padB, _output->getNbElements());
    return _output;
  }

  std::string ReflectionPadding::print() const
  {
    return "ReflectionPadding [" + std::to_string(_padL) + "/" + std::to_string(_padR) + "/" + std::to_string(_padT) + "/" + std::to_string(_padB) + "]";
  }
}
