#include "OpenCLFuncs.h"
#include "SpatialUpsamplingNearest.h"

namespace NN
{
  SpatialUpsamplingNearest::SpatialUpsamplingNearest()
  {
    _upscaleFactor = 2;
  }

  std::shared_ptr<Tensor> SpatialUpsamplingNearest::forward(std::shared_ptr<Tensor> const &input)
  {
    if (input->getSizes().size() != 3)
      throw std::runtime_error("SpatialUpsamplingNearest : Wrong input layout " + input->print());
    std::vector<int> inputSizes = input->getSizes();
    std::vector<int> outputSizes(3);
    outputSizes[0] = input->getSize(0);
    outputSizes[1] = input->getSize(1) * _upscaleFactor;
    outputSizes[2] = input->getSize(2) * _upscaleFactor;

    if (_output == nullptr || _output->getSizes() != outputSizes)
      _output = std::make_shared<Tensor>(outputSizes);

    OpenCLFuncs::getInstance()->spatialUpsamplingNearest(*input, *_output, _upscaleFactor, _output->getNbElements());
    return _output;
  }

  std::string SpatialUpsamplingNearest::print() const
  {
    return "SpatialUpsamplingNearest";
  }

}
