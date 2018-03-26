#ifndef __NODE_H__
#define __NODE_H__

#include <list>
#include "ComputeGraph.h"
#include "Module.h"
#include "Tensor.h"

namespace NN
{
  class Node
  {
  public:
    Node();
    Node(std::list<std::string> const &inputs);
    Node(std::list<std::string> const &inputs, std::shared_ptr<Module> module);

    virtual std::shared_ptr<Tensor> evaluate(ComputeGraph const &graph);

  protected:
    std::list<std::string> _inputs;
    std::shared_ptr<Module> _module;
  };
}

#endif /* __NODE_H__ */
