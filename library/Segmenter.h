#ifndef __SEGMENTER_H__
#define __SEGMENTER_H__

#include "Sequential.h"

namespace NN
{
  class Segmenter
  {
  public:
    Segmenter(std::string const &networkFile);

    std::shared_ptr<Tensor> createIndexesMask(std::shared_ptr<Tensor> input) const;
    std::shared_ptr<Tensor> createRGBMask(std::shared_ptr<Tensor> input) const;
    std::shared_ptr<Tensor> indexesMaskToRGB(std::shared_ptr<Tensor> input) const;

  protected:
    std::shared_ptr<Tensor> extractFeatureMaps(std::shared_ptr<Tensor> input) const;

  protected:
    std::shared_ptr<Sequential> _network;
    std::vector<int> _layers;
    std::vector<int> _weights;
  };
}

#endif /* __SEGMENTER_H__ */
