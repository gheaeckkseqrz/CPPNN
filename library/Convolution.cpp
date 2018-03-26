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

  std::shared_ptr<Input> Convolution::forward(std::shared_ptr<Input> const &input)
  {
    std::shared_ptr<Tensor> inputTensor = std::dynamic_pointer_cast<Tensor>(input);
    if (_filter == nullptr)
      throw std::runtime_error("Convolution isn't setup");
    if (inputTensor == nullptr || inputTensor->getSizes().size() != 3)
      throw std::runtime_error("Convolution recieved invalid input");
    if (_filter->getSize(1) != inputTensor->getSize(0))
	throw std::runtime_error("Convolution recieved invalid input for filter");
    std::vector<int> outputSizes(3, 0);
    outputSizes[0] = _filter->getSize(0);
    outputSizes[1] = (int)floor((float)((inputTensor->getSize(1) + 2 * _padH - _dilationH * (_filter->getSize(2) - 1) - 1) / _dH)) + 1;
    outputSizes[2] = (int)floor((float)((inputTensor->getSize(2) + 2 * _padW - _dilationW * (_filter->getSize(2) - 1) - 1) / _dW)) + 1;

    if (_output == nullptr || std::dynamic_pointer_cast<Tensor>(_output)->getSizes() != outputSizes)
      _output = std::make_shared<Tensor>(outputSizes);

    std::shared_ptr<Tensor> outputTensor = std::dynamic_pointer_cast<Tensor>(_output);
    outputTensor->fill(0);

    std::vector<int> im2colSizes({(int)(inputTensor->getSize(0) * _filter->getSize(2) * _filter->getSize(3)), (int)(outputTensor->getSize(1) * outputTensor->getSize(2))});
    std::shared_ptr<Tensor> im2col = std::make_shared<Tensor>(im2colSizes);

    OpenCLFuncs::getInstance()->convolutionImg2Cols(*inputTensor, *im2col, _filter->getSize(2), _filter->getSize(3), _padW, _padH, _dilationW, _dilationH, im2col->getNbElements());
    _filter->flatten();
    outputTensor->flatten();

    std::shared_ptr<Tensor> tA = _filter;
    std::shared_ptr<Tensor> tB = im2col;
    std::shared_ptr<Tensor> tC = outputTensor;

    cl_command_queue queue = OpenCL::getInstance()->getQueue()();
    int M = tA->getSize(0);
    int N = tB->getSize(1);
    int K = tA->getSize(1);
    clock_t begin = clock();
    cl_event event = NULL;
    int err = clblasSgemm(clblasRowMajor, clblasNoTrans, clblasNoTrans, M, N, K,
			  1,
			  tA->getBuffer()(), tA->getOffset(), tA->getSize(1),
			  tB->getBuffer()(), tB->getOffset(), tB->getSize(1), 1,
			  tC->getBuffer()(), tC->getOffset(), tC->getSize(1),
			  1, &queue, 0, NULL, &event);
    clWaitForEvents(1, &event);
    clReleaseEvent(event);
    clock_t end = clock();
    if (err != CL_SUCCESS)
    	printf("clblasSgemmEx() failed with %d\n", err);
    std::cout << "Running clblasSgemm " << double(end - begin) / CLOCKS_PER_SEC << " sec" << std::endl;

    outputTensor->setSizes(outputSizes);
    if (_bias != nullptr)
      outputTensor->add(*_bias);
    return _output;
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
