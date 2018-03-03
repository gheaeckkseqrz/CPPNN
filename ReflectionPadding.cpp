#include "OpenCLFuncs.h"
#include "ReflectionPadding.h"
#include "Tensor.h"

namespace NN
{
  ReflectionPadding::ReflectionPadding()
  {
  }

  std::shared_ptr<Input> ReflectionPadding::forward(std::shared_ptr<Input> const &input)
  {
    std::shared_ptr<Tensor> inputTensor = std::dynamic_pointer_cast<Tensor>(input);
    assert(inputTensor->getSizes().size() == 3);
    std::vector<int> outputSizes(3, 0);
    outputSizes[0] = inputTensor->getSize(0);
    outputSizes[1] = inputTensor->getSize(1) + _padT + _padB;
    outputSizes[2] = inputTensor->getSize(2) + _padL + _padR;

    if (_output == nullptr || std::dynamic_pointer_cast<Tensor>(_output)->getSizes() != outputSizes)
      {
	_output.reset(new Tensor(outputSizes));
      }
    std::shared_ptr<Tensor> outputTensor = std::dynamic_pointer_cast<Tensor>(_output);
    OpenCLFuncs::getInstance()->reflectionPadding(*(inputTensor.get()), *(outputTensor.get()), _padL, _padR, _padT, _padB, outputTensor->getNbElements());
    return _output;
  }
}
