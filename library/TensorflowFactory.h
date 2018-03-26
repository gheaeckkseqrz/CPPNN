#ifndef __TENSORFLOW_FACTORY_H__
#define __TENSORFLOW_FACTORY_H__

#include <memory>
#include "Node.h"

namespace NN
{
  class TensorflowFactory
  {
  public:
    static TensorflowFactory *getInstance();

    std::shared_ptr<Node> createNode(std::string const &s);

  private:
    TensorflowFactory();
    static TensorflowFactory *_instance;
  };
}

#endif /* __TENSORFLOW_FACTORY_H__ */
