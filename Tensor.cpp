#include "Tensor.h"
#include "OpenCL.h"
#include "OpenCLFuncs.h"

namespace NN
{

  Tensor::Tensor()
  {
    setSizes(std::vector<int>(1, 0));
    _offset = 0;
  }

  Tensor::Tensor(std::vector<int> const &sizes)
  {
    setSizes(sizes);
    _storage = std::make_shared<Storage<float>>(getNbElements());
    _offset = 0;
  }

  Tensor::Tensor(std::vector<float> const &data)
  {
    _storage = std::make_shared<Storage<float>>(data);
    setSizes(std::vector<int>(1, data.size()));
    _offset = 0;
  }

  Tensor::~Tensor()
  {
  }

  size_t Tensor::getSize(int index) const
  {
    return _sizes[index];
  }

  std::vector<int> const &Tensor::getSizes() const
  {
    return _sizes;
  }

  void Tensor::setSizes(std::vector<int> const &sizes)
  {
    _sizes = sizes;
    _sizesStorage = std::unique_ptr<Storage<int>>(new Storage<int>(sizes));
  }

  std::vector<float> Tensor::read() const
  {
    return _storage->read();
  }

  cl::Buffer Tensor::getBuffer() const
  {
    return _storage->getBuffer();
  }

  cl::Buffer Tensor::getSizesBuffer() const
  {
    return _sizesStorage->getBuffer();
  }

  size_t Tensor::getOffset() const
  {
    return _offset;
  }

  size_t Tensor::getNbElements() const
  {
    if (_sizes.size() == 0)
      return 0;
    size_t total = 1;
    for (int s : _sizes)
      total *= s;
    return total;
  }

  Tensor &Tensor::add(float value)
  {
    OpenCLFuncs::getInstance()->tensorValueAdd(*this, *this, value, getNbElements());
    return *this;
  }

  Tensor &Tensor::sub(float value)
  {
    OpenCLFuncs::getInstance()->tensorValueSub(*this, *this, value, getNbElements());
    return *this;
  }

  Tensor &Tensor::mul(float value)
  {
    OpenCLFuncs::getInstance()->tensorValueMul(*this, *this, value, getNbElements());
    return *this;
  }

  Tensor &Tensor::div(float value)
  {
    OpenCLFuncs::getInstance()->tensorValueDiv(*this, *this, value, getNbElements());
    return *this;
  }

  Tensor &Tensor::add(Tensor const &o)
  {
    if (o.getNbElements() == _sizes[0])
      OpenCLFuncs::getInstance()->tensorChannelAdd(*this, o, getNbElements());
    else
      OpenCLFuncs::getInstance()->tensorElementWiseAdd(*this, o, getNbElements());
    return *this;
  }

  Tensor &Tensor::sub(Tensor const &o)
  {
    OpenCLFuncs::getInstance()->tensorElementWiseSub(*this, o, getNbElements());
    return *this;
  }

  Tensor &Tensor::mul(Tensor const &o)
  {
    OpenCLFuncs::getInstance()->tensorElementWiseMul(*this, o, getNbElements());
    return *this;
  }

  Tensor &Tensor::div(Tensor const &o)
  {
    OpenCLFuncs::getInstance()->tensorElementWiseDiv(*this, o, getNbElements());
    return *this;
  }

  Tensor &Tensor::fill(float value)
  {
    OpenCLFuncs::getInstance()->tensorFill(*this, value, getNbElements());
    return *this;
  }

  std::ostream &operator<<(std::ostream &s, Tensor const &t)
  {
    s << "Tensor [";
    std::vector<int> sizes = t.getSizes();
    for (int i(0) ; i < sizes.size() ; ++i)
      s << sizes[i] << ((i < sizes.size() - 1) ? ", " : "");
    s << "] - Offset : " << t.getOffset();
    return s;
  }

}
