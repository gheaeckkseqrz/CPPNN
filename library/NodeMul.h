#ifndef __NODE_MUL_H__
#define __NODE_MUL_H__

#include "Node.h"

namespace NN
{
  class NodeMul : public Node
  {
  public:
    NodeMul(std::list<std::string> const &inputs);

    virtual std::shared_ptr<Tensor> evaluate(ComputeGraph const &graph);
  };
}

#endif /* __NODE_MUL_H__ */
