#include "Convolution.h"
#include "OpenCLFuncs.h"
#include "TensorflowConvolution.h"

namespace NN
{
  TensorflowConvolution::TensorflowConvolution(std::list<std::string> const &inputs)
    :Node(inputs)
  {
    _module = std::make_shared<Convolution>();
  }

  std::shared_ptr<Tensor> TensorflowConvolution::evaluate(ComputeGraph const &graph)
  {
    std::shared_ptr<Tensor> input = graph.getNode(_inputs.front())->evaluate(graph);
    std::shared_ptr<Tensor> filter = graph.getNode(_inputs.back())->evaluate(graph);

    std::dynamic_pointer_cast<Convolution>(_module)->setFilter(flipFilter(filter));
    NN::Input *dummy = nullptr; // Make sure to fix next line when removing Input
    return std::dynamic_pointer_cast<Tensor>(_module->forward(input));
  }

  std::shared_ptr<Tensor> TensorflowConvolution::flipFilter(std::shared_ptr<Tensor> const &filter)
  {
    std::vector<int> newSizes({filter->getSize(3), filter->getSize(2), filter->getSize(0), filter->getSize(1)});
    std::shared_ptr<Tensor> newFilter = std::make_shared<Tensor>(newSizes);
    OpenCLFuncs::getInstance()->convolutionChangeFilterPacking(*filter, *newFilter, newFilter->getNbElements());
    return newFilter;
  }
}
