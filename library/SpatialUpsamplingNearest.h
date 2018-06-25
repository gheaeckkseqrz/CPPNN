#ifndef __SPATIALUPSAMPLINGNEAREST_H__
#define __SPATIALUPSAMPLINGNEAREST_H__

#include "Module.h"

namespace NN
{
  class SpatialUpsamplingNearest : public Module
  {
  public:
    SpatialUpsamplingNearest();

    virtual std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> const &input);
    virtual std::string print() const;

  protected:
    int _upscaleFactor;
  };
}

#endif /* __SPATIALUPSAMPLINGNEAREST_H__ */
