#include "ParametricModel.h"

namespace NN
{
  ParametricModel::ParametricModel()
  {
  }

  ParametricModel::ParametricModel(std::vector<float> &relu1_1,
				   std::vector<float> &relu2_1,
				   std::vector<float> &relu3_1,
				   std::vector<float> &relu4_1,
				   std::vector<float> &relu5_1)
    :_relu1_1(relu1_1), _relu2_1(relu2_1), _relu3_1(relu3_1), _relu4_1(relu4_1), _relu5_1(relu5_1)
  {
  }

  std::vector<float> ParametricModel::getFullModel() const
  {
    std::vector<float> fullModel;
    fullModel.reserve(MODEL_SIZE);
    fullModel.insert(fullModel.end(), _relu1_1.begin(), _relu1_1.end());
    fullModel.insert(fullModel.end(), _relu2_1.begin(), _relu2_1.end());
    fullModel.insert(fullModel.end(), _relu3_1.begin(), _relu3_1.end());
    fullModel.insert(fullModel.end(), _relu4_1.begin(), _relu4_1.end());
    fullModel.insert(fullModel.end(), _relu5_1.begin(), _relu5_1.end());
    return fullModel;
  }
}
