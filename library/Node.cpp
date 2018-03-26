#include "Node.h"

namespace NN
{
  Node::Node()
  {
  }

  Node::Node(std::list<std::string> const &inputs)
    :_inputs(inputs)
  {
  }

  Node::Node(std::list<std::string> const &inputs, std::shared_ptr<Module> module)
    :_inputs(inputs), _module(module)
  {
  }

  std::shared_ptr<Tensor> Node::evaluate(ComputeGraph const &graph)
  {
    if (_module == nullptr)
      return graph.getNode(_inputs.front())->evaluate(graph);
    if (_inputs.empty())
      return std::dynamic_pointer_cast<Tensor>(_module->forward(nullptr));
    if (_inputs.size() == 1)
      return std::dynamic_pointer_cast<Tensor>(_module->forward(graph.getNode(_inputs.front())->evaluate(graph)));
    else
      std::cerr << "Basic Node base class doesn't support multiple inputs" << std::endl;
    return nullptr;
  }
}
