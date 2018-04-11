#include <clBLAS.h>
#include <cmath>
#include <ctime>
#include <iostream>
#include <stdexcept>

#include "Convolution.h"
#include "OpenCLFuncs.h"

namespace NN
{
  Convolution::Convolution()
  {
    _padW = 1;
    _padH = 1;
    _dW = 1;
    _dH = 1;
    _dilationW = 1;
    _dilationH = 1;
  }

  Convolution::~Convolution()
  {
  }

  void Convolution::setFilter(std::shared_ptr<Tensor> const &filter, std::shared_ptr<Tensor> const &bias)
  {
    _filter = filter;
    _bias = bias;
  }

  void Convolution::setPadding(int padW, int padH)
  {
    _padW = padW;
    _padH = padH;
  }

  std::shared_ptr<Tensor> Convolution::forward(std::shared_ptr<Tensor> const &input)
  {
    if (_filter == nullptr)
      throw std::runtime_error("Convolution isn't setup");
    if (input == nullptr || input->getSizes().size() != 3)
      throw std::runtime_error("Convolution recieved invalid input");
    if (_filter->getSize(1) != input->getSize(0))
	throw std::runtime_error("Convolution recieved invalid input for filter");

    std::cout << "CONV " << _filter->getSize(1) << " -> " << _filter->getSize(0) << std::endl;

    std::vector<int> filterSizes = _filter->getSizes();
    std::vector<int> outputSizes(3, 0);
    outputSizes[0] = _filter->getSize(0);
    outputSizes[1] = (int)floor((float)((input->getSize(1) + 2 * _padH - _dilationH * (_filter->getSize(2) - 1) - 1) / _dH)) + 1;
    outputSizes[2] = (int)floor((float)((input->getSize(2) + 2 * _padW - _dilationW * (_filter->getSize(2) - 1) - 1) / _dW)) + 1;

    if (_output == nullptr || _output->getSizes() != outputSizes)
      _output = std::make_shared<Tensor>(outputSizes);
    _output->fill(0);

    std::vector<int> im2colSizes({(int)(input->getSize(0) * _filter->getSize(2) * _filter->getSize(3)), (int)(_output->getSize(1) * _output->getSize(2))});
    std::shared_ptr<Tensor> im2col = std::make_shared<Tensor>(im2colSizes);

    _filter->flatten();
    _output->flatten();

    for (int i(0) ; i < outputSizes[0] ; ++i)
      {
	processBlock(input, im2col, std::pair<int, int>(i, i+1), filterSizes[2], filterSizes[3]);
      }

    _output->setSizes(outputSizes);
    _filter->setSizes(filterSizes);
    if (_bias != nullptr)
      _output->add(*_bias);
    return _output;
  }

  void Convolution::processBlock(std::shared_ptr<Tensor> input, std::shared_ptr<Tensor> im2col, std::pair<int, int> slice, int kernelH, int kernelW)
  {
    std::shared_ptr<Tensor> outputSlice = (*_output)[slice];
    std::shared_ptr<Tensor> filterSlice = (*_filter)[slice];
    OpenCLFuncs::getInstance()->convolutionImg2Cols(*input, *im2col, kernelH, kernelW, _padW, _padH, _dilationW, _dilationH, im2col->getNbElements());
    filterSlice->matrixMultiply(*im2col, outputSlice);
  }

  std::string Convolution::print() const
  {
    if (_filter == nullptr)
      return "Convolution - No filter";
    std::string s = "Convolution " + std::to_string(_filter->getSize(1)) + " -> " + std::to_string(_filter->getSize(0));
    s += " [" + std::to_string(_filter->getSize(3)) + "x" + std::to_string(_filter->getSize(2)) + "]";
    s += " Padding [" + std::to_string(_padW) + "/" + std::to_string(_padH) + "]";
    s += " Stride [" + std::to_string(_dW) + "/" + std::to_string(_dH) + "]";
    s += " Dilation [" + std::to_string(_dilationW) + "/" + std::to_string(_dilationH) + "]";
    return s;
  }
}
