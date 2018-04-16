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
}
