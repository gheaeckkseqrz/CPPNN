#include "MatToTensor.h"
#include "OpenCLFuncs.h"

std::shared_ptr<NN::Tensor> matToTensor(cv::Mat const &m)
{
  cv::Mat m2;
  m.convertTo(m2, CV_32FC3);
  std::vector<float> v;
  v.assign((float*)m2.datastart, (float*)m2.dataend);
  NN::Tensor badLayout(std::vector<int>({3, m2.cols, m2.rows}), v);
  std::shared_ptr<NN::Tensor> goodLayout = std::make_shared<NN::Tensor>(std::vector<int>({3, m.cols, m.rows}));
  NN::OpenCLFuncs::getInstance()->tensorFromMat(badLayout, *goodLayout, goodLayout->getNbElements());
  return goodLayout;
}

std::shared_ptr<NN::Tensor> tensorFromImage(std::string const &path)
{
  cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
  if(!image.data)
    {
      std::cerr << "Could not open or find the image [" << path << "]" << std::endl;
      return nullptr;
    }
  return matToTensor(image);
}