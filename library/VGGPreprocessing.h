#ifndef __VGGPREPROCESSING_H__
#define __VGGPREPROCESSING_H__

#include "Convolution.h"

namespace NN
{
  class VGGPreprocessing : public Convolution
  {
  public:
    VGGPreprocessing();
    virtual std::string print() const;
  };
}

#endif /* __VGGPREPROCESSING_H__ */
