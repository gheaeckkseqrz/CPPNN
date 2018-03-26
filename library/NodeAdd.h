#ifndef __NODE_ADD_H__
#define __NODE_ADD_H__

#include "Node.h"

namespace NN
{
  class NodeAdd : public Node
  {
  public:
    NodeAdd(std::list<std::string> const &inputs);

    virtual std::shared_ptr<Tensor> evaluate(ComputeGraph const &graph);
  };

}

#endif /* __ADD_H__ */
