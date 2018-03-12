#include "OpenCL.h"

namespace NN
{
  // template<class T>
  // Storage<T>::Storage(cl_mem_flags flags)
  // {
  //   _size = 0;
  //   _buffer = 0;
  // }

  template<class T>
  Storage<T>::Storage(size_t size, cl_mem_flags flags)
  {
    _size = size;
    _buffer = 0;
    if (_size > 0)
      _buffer = OpenCL::getInstance()->createBuffer<T>(_size);
  }

  template<class T>
  Storage<T>::Storage(std::vector<T> const &data, cl_mem_flags flags)
  {
    _buffer = OpenCL::getInstance()->toGPU<T>(data);
    _size = data.size();
  }

  template<class T>
  cl::Buffer Storage<T>::getBuffer() const
  {
    return _buffer;
  }

  template<class T>
  std::vector<T> Storage<T>::read() const
  {
    std::vector<T> b(_size);
    OpenCL::getInstance()->fromGPU<T>(b, _buffer);
    return b;
  }
}
