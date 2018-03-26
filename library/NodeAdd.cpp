#include "NodeAdd.h"
#include "Tensor.h"

namespace NN
{
  NodeAdd::NodeAdd(std::list<std::string> const &inputs)
    :Node(inputs)
  {
  }

  std::shared_ptr<Tensor> NodeAdd::evaluate(ComputeGraph const &graph)
  {
    std::shared_ptr<Tensor> output(nullptr);
    for (std::string input : _inputs)
      {
	if (output == nullptr)
	  output = graph.getNode(input)->evaluate(graph); // ->clone();
	else
	  output->add(*(graph.getNode(input)->evaluate(graph)));
      }
    return output;
  }
}
