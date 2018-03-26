#include "NodeMul.h"
#include "Tensor.h"

namespace NN
{
  NodeMul::NodeMul(std::list<std::string> const &inputs)
    :Node(inputs)
  {
  }

  std::shared_ptr<Tensor> NodeMul::evaluate(ComputeGraph const &graph)
  {
    std::shared_ptr<Tensor> output(nullptr);
    for (std::string input : _inputs)
      {
	if (output == nullptr)
	  output = graph.getNode(input)->evaluate(graph); // ->clone();
	else
	  output->mul(*(graph.getNode(input)->evaluate(graph)));
      }
    return output;
  }

}
