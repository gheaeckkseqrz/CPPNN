#ifndef __STORAGE_H__
#define __STORAGE_H__

#include "cl.hpp"

namespace NN
{

  template<class T> class Storage
  {
  public:
    /* Storage(cl_mem_flags flags =  CL_MEM_READ_WRITE ); */
    Storage(size_t size, cl_mem_flags flags =  CL_MEM_READ_WRITE );
    Storage(std::vector<T> const &data, cl_mem_flags flags =  CL_MEM_READ_WRITE );
    virtual ~Storage() { }

    cl::Buffer getBuffer() const;
    std::vector<T> read(size_t offset = 0, size_t size = -1) const;
    void write(std::vector<T> const &data, size_t offset);

  protected:
    cl::Buffer _buffer;
    size_t _size;
  };
}

#include "Storage.cpp"

#endif
