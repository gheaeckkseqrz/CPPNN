#include "ParametricModel.h"

namespace NN
{
  ParametricModel::ParametricModel()
  {
  }

  ParametricModel::ParametricModel(std::vector<float> &fullModel)
  {
    int pos = 0;
    _relu1_1 = std::vector<float>(fullModel.begin() + pos, fullModel.begin() + pos + getModelSize(e_relu1_1));
    pos += getModelSize(e_relu1_1);
    _relu1_1 = std::vector<float>(fullModel.begin() + pos, fullModel.begin() + pos + getModelSize(e_relu2_1));
    pos += getModelSize(e_relu2_1);
    _relu1_1 = std::vector<float>(fullModel.begin() + pos, fullModel.begin() + pos + getModelSize(e_relu3_1));
    pos += getModelSize(e_relu3_1);
    _relu1_1 = std::vector<float>(fullModel.begin() + pos, fullModel.begin() + pos + getModelSize(e_relu4_1));
    pos += getModelSize(e_relu4_1);
    _relu1_1 = std::vector<float>(fullModel.begin() + pos, fullModel.begin() + pos + getModelSize(e_relu5_1));
  }

  ParametricModel::ParametricModel(std::vector<float> &relu1_1,
				   std::vector<float> &relu2_1,
				   std::vector<float> &relu3_1,
				   std::vector<float> &relu4_1,
				   std::vector<float> &relu5_1)
    :_relu1_1(relu1_1), _relu2_1(relu2_1), _relu3_1(relu3_1), _relu4_1(relu4_1), _relu5_1(relu5_1)
  {
  }

  std::vector<float> ParametricModel::getFullModel(int components) const
  {
    std::vector<float> fullModel;
    fullModel.reserve(getModelSize(components));
    if (components & e_relu1_1)
      fullModel.insert(fullModel.end(), _relu1_1.begin(), _relu1_1.end());
    if (components & e_relu2_1)
      fullModel.insert(fullModel.end(), _relu2_1.begin(), _relu2_1.end());
    if (components & e_relu3_1)
      fullModel.insert(fullModel.end(), _relu3_1.begin(), _relu3_1.end());
    if (components & e_relu4_1)
      fullModel.insert(fullModel.end(), _relu4_1.begin(), _relu4_1.end());
    if (components & e_relu5_1)
      fullModel.insert(fullModel.end(), _relu5_1.begin(), _relu5_1.end());
    return fullModel;
  }

  int ParametricModel::getModelSize(int components)
  {
    int s = 0;
    if (components & e_relu1_1) s += (64 * 64 + 64) / 2;
    if (components & e_relu2_1) s += (128 * 128 + 128) / 2;
    if (components & e_relu3_1) s += (256 * 256 + 256) / 2;
    if (components & e_relu4_1) s += (512 * 512 + 512) / 2;
    if (components & e_relu5_1) s += (512 * 512 + 512) / 2;
    return s;
  }

}
