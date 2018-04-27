#ifndef __PARAMETRIC_MODEL_H__
#define __PARAMETRIC_MODEL_H__

#include <string>
#include <vector>

#define FULL_MODEL (ParametricModel::e_relu1_1 | ParametricModel::e_relu2_1 | ParametricModel::e_relu3_1 | ParametricModel::e_relu4_1 | ParametricModel::e_relu5_1)

namespace NN
{
  class ParametricModel
  {
  public:
    enum
    {
      e_relu1_1 = 1,
      e_relu2_1 = 1 << 1,
      e_relu3_1 = 1 << 2,
      e_relu4_1 = 1 << 3,
      e_relu5_1 = 1 << 4,
    };

  public:
    ParametricModel();
    ParametricModel(std::vector<float> &relu1_1,
		    std::vector<float> &relu2_1,
		    std::vector<float> &relu3_1,
		    std::vector<float> &relu4_1,
		    std::vector<float> &relu5_1);

    std::vector<float> getFullModel(int components = FULL_MODEL) const;
    static int getModelSize(int components);

  private:
    std::vector<float> _relu1_1;
    std::vector<float> _relu2_1;
    std::vector<float> _relu3_1;
    std::vector<float> _relu4_1;
    std::vector<float> _relu5_1;
  };
}

#endif /* __PARAMETRIC_MODEL_H__ */
