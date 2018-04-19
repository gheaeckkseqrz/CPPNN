#ifndef __PARAMETRIC_MODEL_H__
#define __PARAMETRIC_MODEL_H__

#include <string>
#include <vector>

//#define MODEL_SIZE ((512 * 512 + 512) / 2 + (512 * 512 + 512) / 2 + (256 * 256 + 256) / 2 + (128 * 128 + 128) / 2 + (64 * 64 + 64) / 2)
#define MODEL_SIZE ((64 * 64 + 64) / 2)
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

    std::vector<float> getFullModel() const;

  private:
    std::vector<float> _relu1_1;
    std::vector<float> _relu2_1;
    std::vector<float> _relu3_1;
    std::vector<float> _relu4_1;
    std::vector<float> _relu5_1;
  };
}

#endif /* __PARAMETRIC_MODEL_H__ */
