#ifndef __TORCHCONOLUTION_H__
#define __TORCHCONOLUTION_H__

#include "Convolution.h"
#include "TorchObject.h"

namespace NN
{
  class TorchConvolution : public Convolution, public TorchObject
  {
  public:
    TorchConvolution();

    virtual std::shared_ptr<TorchObject> loadFromFile(std::ifstream &file, std::map<int, std::shared_ptr<TorchObject>> &loaded);
  };
}

#endif /* __TORCHCONOLUTION_H__ */
