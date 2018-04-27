#ifndef __MATTOTENSOR__
#define __MATTOTENSOR__

#include <opencv2/opencv.hpp>
#include "Tensor.h"

std::shared_ptr<NN::Tensor> matToTensor(cv::Mat const &m);
std::shared_ptr<NN::Tensor> tensorFromImage(std::string const &path, int maxSize = -1, int border = 0);
void saveTensorAsImage(std::shared_ptr<NN::Tensor> t, std::string const &path);

#endif
