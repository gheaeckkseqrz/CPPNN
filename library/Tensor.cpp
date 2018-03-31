#include <algorithm>
#include "Tensor.h"
#include "OpenCL.h"
#include "OpenCLFuncs.h"

namespace NN
{

  Tensor::Tensor(cl_mem_flags flags)
  {
    setSizes(std::vector<int>(1, 0));
    _offset = 0;
  }

  Tensor::Tensor(std::vector<int> const &sizes, cl_mem_flags flags)
  {
    setSizes(sizes);
    _storage = std::make_shared<Storage<float>>(getNbElements(), flags);
    _offset = 0;
  }

  Tensor::Tensor(std::vector<float> const &data, cl_mem_flags flags)
  {
    _storage = std::make_shared<Storage<float>>(data, flags);
    setSizes(std::vector<int>(1, data.size()));
    _offset = 0;
  }

  Tensor::Tensor(std::vector<int> const &sizes, std::vector<float> const &data, cl_mem_flags flags)
  {
    setSizes(sizes);
    if (getNbElements() != data.size())
      throw std::runtime_error("Tensor Constructor with invalid values");
    _storage = std::make_shared<Storage<float>>(data);
    _offset = 0;
  }

  Tensor::Tensor(Tensor const &o)
  {
    setSizes(o.getSizes());
    _storage = o._storage;
    _offset = o._offset;
    std::cout << "[WARN] - Calling Tensor copy constructor" << std::endl;
  }

  Tensor::~Tensor()
  {
  }

  int Tensor::getSize(int index) const
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

  void Tensor::flatten()
  {
    std::vector<int> newSizes;
    newSizes.push_back(getSize(0));
    newSizes.push_back(getNbElements() / getSize(0));
    setSizes(newSizes);
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

  bool Tensor::dataEquals(Tensor const &o, float tolerance) const
  {
    if (getNbElements() != o.getNbElements())
      return false;
    Tensor output(std::vector<float>({0}));
    OpenCLFuncs::getInstance()->tensorDataEquals(*this, o, output, tolerance, getNbElements());
    std::vector<float> res = output.read();
    return res[0] == 0;
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
      OpenCLFuncs::getInstance()->tensorChannelAdd(*this, *this, o, getNbElements());
    else
      OpenCLFuncs::getInstance()->tensorElementWiseAdd(*this, *this, o, getNbElements());
    return *this;
  }

  Tensor &Tensor::sub(Tensor const &o)
  {
    if (o.getNbElements() == _sizes[0])
      OpenCLFuncs::getInstance()->tensorChannelSub(*this, *this, o, getNbElements());
    else
      OpenCLFuncs::getInstance()->tensorElementWiseSub(*this, *this, o, getNbElements());
    return *this;
  }

  Tensor &Tensor::mul(Tensor const &o)
  {
    if (o.getNbElements() == _sizes[0])
      OpenCLFuncs::getInstance()->tensorChannelMul(*this, *this, o, getNbElements());
    else
      OpenCLFuncs::getInstance()->tensorElementWiseMul(*this, *this, o, getNbElements());
    return *this;
  }

  Tensor &Tensor::div(Tensor const &o)
  {
    if (o.getNbElements() == _sizes[0])
      OpenCLFuncs::getInstance()->tensorChannelDiv(*this, *this, o, getNbElements());
    else
      OpenCLFuncs::getInstance()->tensorElementWiseDiv(*this, *this, o, getNbElements());
    return *this;
  }

  Tensor &Tensor::fill(float value)
  {
    OpenCLFuncs::getInstance()->tensorFill(*this, value, getNbElements());
    return *this;
  }

  Tensor &Tensor::copy(Tensor const &o)
  {
    assert(getNbElements() == o.getNbElements());
    OpenCLFuncs::getInstance()->tensorCopy(*this, o, getNbElements());
    return *this;
  }

  Tensor Tensor::means() const
  {
    Tensor res(std::vector<int>({getSize(0)}));
    OpenCLFuncs::getInstance()->tensorMeans(*this, res, res.getNbElements());
    return res;
  }

  float Tensor::min() const
  {
    std::vector<float> vals = read();
    return *std::min_element(vals.begin(), vals.end());
  }

  float Tensor::max() const
  {
    std::vector<float> vals = read();
    return *std::max_element(vals.begin(), vals.end());
  }

  Tensor Tensor::transpose() const
  {
    assert(getSizes().size() == 2);
    Tensor res(std::vector<int>({getSize(1), getSize(0)}));
    OpenCLFuncs::getInstance()->tensorTranspose(*this, res, res.getNbElements());
    return res;
  }

  std::string Tensor::print(bool data) const
  {
    std::string s = "Tensor [";
    std::vector<int> sizes = getSizes();
    for (int i(0) ; i < sizes.size() ; ++i)
      s += std::to_string(sizes[i]) + ((i < sizes.size() - 1) ? ", " : "");
    s += "] - Offset : " + std::to_string(getOffset());
    s += " || Range [" + std::to_string(min()) + ",  " + std::to_string(max()) + "]";

      if (data)
      {
	s += "\n{";
	std::vector<float> d = read();
	for (int i(0) ; i < d.size() ; ++i)
	  {
	    if ((i % (getNbElements() / getSize(0))) == 0 && _sizes.size() > 1)
	      s += '\n';
	    s += std::to_string(d[i]).substr(0, 5) + ", ";
	  }
	s += "\n}\n";
      }

    return s;
  }

  Tensor Tensor::operator[](int index)
  {
    std::vector<int> sizes(_sizes.begin() + 1, _sizes.end());
    Tensor ret = Tensor(sizes);
    ret._storage = _storage;
    ret._offset = _offset + (index * ret.getNbElements());
    return ret;
  }

  Tensor Tensor::operator[](std::pair<int, int> range)
  {
    std::vector<int> sizes(_sizes);
    sizes[0] = range.second - range.first;
    Tensor ret = Tensor(sizes);
    ret._storage = _storage;
    ret._offset = _offset + (range.first * (getNbElements() / _sizes[0]));
    return ret;
  }

  std::ostream &operator<<(std::ostream &s, Tensor const &t)
  {
    s << t.print();
    return s;
  }

}
