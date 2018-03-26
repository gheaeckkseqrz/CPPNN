#include "Placeholder.h"

namespace NN
{
  Placeholder::Placeholder()
  {
  }

  Placeholder::Placeholder(std::shared_ptr<Tensor> content)
    :_content(content)
  {
  }

  std::shared_ptr<Tensor> Placeholder::evaluate(ComputeGraph const &graph)
  {
    return _content;
  }

  void Placeholder::setContent(std::shared_ptr<Tensor> content)
  {
    _content = content;
  }
}
