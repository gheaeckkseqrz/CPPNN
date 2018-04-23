#ifndef __OPEN_CL_H__
#define __OPEN_CL_H__

#include <iostream>
#include <assert.h>
#include "cl.hpp"

class OpenCL
{
 private:
  OpenCL();

 public:
  static OpenCL *getInstance();
  ~OpenCL();

  template<typename T> cl::Buffer createBuffer(size_t size)
    {
      return cl::Buffer(_context, CL_MEM_READ_WRITE, size * sizeof(T));
    }

  template<typename T> cl::Buffer toGPU(std::vector<T> const &data)
    {
      size_t size = data.size() * sizeof(T);
      cl::Buffer b(_context, CL_MEM_READ_WRITE, size);
      assert(_queue.enqueueWriteBuffer(b, CL_TRUE, 0, size, data.data()) == CL_SUCCESS);
      return b;
    }

  template<typename T> void fromGPU(std::vector<T> &data, cl::Buffer const &b, size_t offset = 0, size_t size = -1)
    {
      if (size == -1)
	size = data.size();
      size *= sizeof(T);
      offset *= sizeof(T);
      assert(_queue.enqueueReadBuffer(b, CL_TRUE, offset, size, data.data()) == CL_SUCCESS);
    }

  template<typename T> void writeToBuffer(cl::Buffer const &b, std::vector<T> const &data, size_t offset = 0)
    {
      size_t size = data.size() * sizeof(T);
      assert(_queue.enqueueWriteBuffer(b, CL_TRUE, offset * sizeof(T), size, data.data()) == CL_SUCCESS);
    }


  cl::Program buildProgramFromSource(std::string const &source);
  cl::Program buildProgramFromSource(std::vector<char> const &source);
  cl::Program buildProgramFromFile(std::string const &path);
  void runKernel(cl::Kernel &kernel, unsigned int workItems, unsigned int groupSize = -1);

  cl::CommandQueue getQueue() const;
  cl::Context getContext() const;

 private:
  cl::Device getOpenCLDevice();
  std::vector<char> getFileContent(std::string const &path);

 private:
  static OpenCL *_instance;
  cl::Context _context;
  cl::Device _device;
  cl::CommandQueue _queue;
  unsigned int _maxGroupSize;
};

#endif
