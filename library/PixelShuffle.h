#ifndef __PIXELSHUFFLE_H__
#define __PIXELSHUFFLE_H__

#include "Module.h"

namespace NN
{
  class PixelShuffle : public Module
  {
  public:
    PixelShuffle();

    virtual std::shared_ptr<Tensor> forward(std::shared_ptr<Tensor> const &input);
    virtual std::string print() const;

  private:
    int _upscaleFactor;
    int _upscaleFactorSquarred;
  };
}

#endif /* __PIXELSHUFFLE_H__ */
