#ifndef __STORAGE_H__
#define __STORAGE_H__

#include "cl.hpp"

namespace NN
{

  template<class T> class Storage
  {
  public:
    Storage();
    Storage(size_t size);
    Storage(std::vector<T> const &data);
    virtual ~Storage() { }

    cl::Buffer getBuffer() const;
    std::vector<T> read() const;

  protected:
    cl::Buffer _buffer;
    size_t _size;
  };
}

#include "Storage.cpp"

#endif
