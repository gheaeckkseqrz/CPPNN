#include "ComputeGraph.h"
#include "Node.h"

namespace NN
{
  ComputeGraph::ComputeGraph()
  {

  }

  void ComputeGraph::add(std::string const &name, std::shared_ptr<Node> node)
  {
    _nodes[name] = node;
  }

  std::shared_ptr<Node> ComputeGraph::getNode(std::string const &nodeName) const
  {
    return _nodes.at(nodeName);
  }
}
