#include "MaxPooling.h"
#include "OpenCLFuncs.h"
#include "Tensor.h"

namespace NN
{
  MaxPooling::MaxPooling()
  {
  }

  std::shared_ptr<Tensor> MaxPooling::forward(std::shared_ptr<Tensor> const &input)
  {
    assert(input->getSizes().size() == 3);
    std::vector<int> outputSizes(3, 0);
    outputSizes[0] = input->getSize(0);
    outputSizes[1] = input->getSize(1) / 2;
    outputSizes[2] = input->getSize(2) / 2;
    if (_output == nullptr || _output->getSizes() != input->getSizes())
	_output.reset(new Tensor(outputSizes));
    OpenCLFuncs::getInstance()->maxPooling(*input, *_output, _output->getNbElements());
    return _output;
  }

  std::string MaxPooling::print() const
  {
    return "MaxPooling";
  }
}
