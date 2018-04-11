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
    std::shared_ptr<Tensor> ret;
    if (_module == nullptr)
      ret = graph.getNode(_inputs.front())->evaluate(graph);
    else if (_inputs.empty())
      ret = _module->forward(nullptr);
    else if (_inputs.size() == 1)
      ret = _module->forward(graph.getNode(_inputs.front())->evaluate(graph));
    else
      std::cerr << "Basic Node base class doesn't support multiple inputs" << std::endl;
    if (_module)
      _module->clearOutput();
    return ret;
  }
}
