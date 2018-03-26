#ifndef __PLACEHOLDER_H__
#define __PLACEHOLDER_H__

#include "Node.h"
#include "Tensor.h"

namespace NN
{
  class Placeholder : public Node
  {
  public:
    Placeholder();
    Placeholder(std::shared_ptr<Tensor> content);
    virtual ~Placeholder() {}

    virtual std::shared_ptr<Tensor> evaluate(ComputeGraph const &graph);
    void setContent(std::shared_ptr<Tensor> content);

  private:
    std::shared_ptr<Tensor> _content;
  };
}

#endif
