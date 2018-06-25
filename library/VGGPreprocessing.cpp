#include "VGGPreprocessing.h"

namespace NN
{
  VGGPreprocessing::VGGPreprocessing()
  {
    std::vector<int> filterSize({3, 3, 1, 1});
    std::vector<float> filterData({0, 0, 1, 0, 1, 0, 1, 0, 0});
    std::shared_ptr<Tensor> filter = std::make_shared<Tensor>(filterSize, filterData);

    std::vector<int> biasSize({3});
    std::vector<float> biasData({-103.9390, -116.7790, -123.6800});
    std::shared_ptr<Tensor> bias = std::make_shared<Tensor>(biasSize, biasData);

    setFilter(filter, bias);
  }

  std::string VGGPreprocessing::print() const
  {
    return "VGGPreprocessing";
  }
}
