#ifndef __COMPUTE_GRAPH__
#define __COMPUTE_GRAPH__

#include <map>
#include <memory>
#include <string>


namespace NN
{
  class Node;

  class ComputeGraph
  {
  public:
    ComputeGraph();

    void add(std::string const &name, std::shared_ptr<Node> node);
    std::shared_ptr<Node> getNode(std::string const &nodeName) const;

  private:
    std::map<std::string, std::shared_ptr<Node>> _nodes;
  };
}

#endif /* __COMPUTE_GRAPH__ */
