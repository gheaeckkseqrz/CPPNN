#include "MatToTensor.h"
#include "OpenCLFuncs.h"

std::shared_ptr<NN::Tensor> matToTensor(cv::Mat const &m)
{
  cv::Mat m2;
  m.convertTo(m2, CV_32FC3);
  std::vector<float> v;
  v.assign((float*)m2.datastart, (float*)m2.dataend);
  NN::Tensor badLayout(std::vector<int>({3, m2.cols, m2.rows}), v);
  std::shared_ptr<NN::Tensor> goodLayout = std::make_shared<NN::Tensor>(std::vector<int>({3, m.rows, m.cols}));
  NN::OpenCLFuncs::getInstance()->tensorFromMat(badLayout, *goodLayout, goodLayout->getNbElements());
  return goodLayout;
}

std::shared_ptr<NN::Tensor> tensorFromImage(std::string const &path, int maxSize, int border)
{
  cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
  if(!image.data)
    {
      std::cerr << "Could not open or find the image [" << path << "]" << std::endl;
      return nullptr;
    }
  if (border > 0)
    {
      if (image.cols < (2 * border + 100) || image.rows < (2 * border + 100))
	{
	  std::cerr << "Image [" << path << "] is too small to be used" << std::endl;
	  return nullptr;
	}
      cv::Rect myROI(border, border, image.cols - (2 * border), image.rows - (2 * border));
      cv::Mat croppedimage = image(myROI);
      image = croppedimage;
    }
  if (maxSize > 0)
    {
      float ratio = (float)maxSize / (float)std::max(image.cols, image.rows);
      cv::resize(image, image, cv::Size(), ratio, ratio);
    }
  return matToTensor(image);
}

void saveTensorAsImage(std::shared_ptr<NN::Tensor> t, std::string const &path)
{
  std::shared_ptr<NN::Tensor> goodLayout = std::make_shared<NN::Tensor>(t->getSizes());
  NN::OpenCLFuncs::getInstance()->tensorToMat(*t, *goodLayout, goodLayout->getNbElements());
  std::vector<float> data = goodLayout->read();
  cv::Mat m(goodLayout->getSize(1), goodLayout->getSize(2), CV_32FC3, data.data());
  try
    {
      cv::imwrite(path, m);
    }
  catch (std::runtime_error &ex)
    {
      std::cerr << "imwrite threw exception : " << ex.what() << std::endl;
    }
}
