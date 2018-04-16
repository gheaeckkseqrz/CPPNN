#ifndef __PARAMETRIC_MODEL_H__
#define __PARAMETRIC_MODEL_H__

#include <string>
#include <vector>

namespace NN
{
  class ParametricModel
  {
  public:
    ParametricModel();
    ParametricModel(std::vector<float> &relu1_1,
		    std::vector<float> &relu2_1,
		    std::vector<float> &relu3_1,
		    std::vector<float> &relu4_1,
		    std::vector<float> &relu5_1);

  private:
    std::vector<float> _relu1_1;
    std::vector<float> _relu2_1;
    std::vector<float> _relu3_1;
    std::vector<float> _relu4_1;
    std::vector<float> _relu5_1;
  };
}

#endif /* __PARAMETRIC_MODEL_H__ */
