#include "MaxPooling.h"
#include "OpenCLFuncs.h"
#include "Tensor.h"

namespace NN
{
  MaxPooling::MaxPooling()
  {
  }

  std::shared_ptr<Input> MaxPooling::forward(std::shared_ptr<Input> const &input)
  {
    std::shared_ptr<Tensor> inputTensor = std::dynamic_pointer_cast<Tensor>(input);
    assert(inputTensor->getSizes().size() == 3);
    std::vector<int> outputSizes(3, 0);
    outputSizes[0] = inputTensor->getSize(0);
    outputSizes[1] = inputTensor->getSize(1) / 2;
    outputSizes[2] = inputTensor->getSize(2) / 2;
    if (_output == nullptr || std::dynamic_pointer_cast<Tensor>(_output)->getSizes() != inputTensor->getSizes())
	_output.reset(new Tensor(outputSizes));
    std::shared_ptr<Tensor> outputTensor = std::dynamic_pointer_cast<Tensor>(_output);
    OpenCLFuncs::getInstance()->maxPooling(*inputTensor, *outputTensor, outputTensor->getNbElements());
    return _output;
  }

  std::string MaxPooling::print() const
  {
    return "MaxPooling";
  }
}
