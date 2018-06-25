#ifndef __TORCHSPATIALUPSAMPLINGNEAREST_H__
#define __TORCHSPATIALUPSAMPLINGNEAREST_H__

#include "SpatialUpsamplingNearest.h"

namespace NN
{
  class TorchSpatialUpsamplingNearest : public SpatialUpsamplingNearest, public TorchObject
  {
  public:
    TorchSpatialUpsamplingNearest();

    virtual std::shared_ptr<TorchObject> loadFromFile(std::ifstream &file, std::map<int, std::shared_ptr<TorchObject>> &loaded);
  };
}

#endif /* __TORCHSPATIALUPSAMPLINGNEAREST_H__ */
