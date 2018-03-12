#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <memory>
#include <vector>
#include "cl.hpp"

#include "Input.h"
#include "Storage.h"

namespace NN
{
  class Tensor : public Input
  {
  public:
    Tensor(cl_mem_flags flags =  CL_MEM_READ_WRITE);
    Tensor(std::vector<int> const &sizes, cl_mem_flags flags =  CL_MEM_READ_WRITE );
    Tensor(std::vector<float> const &data, cl_mem_flags flags =  CL_MEM_READ_WRITE );
    Tensor(std::vector<int> const &sizes, std::vector<float> const &data, cl_mem_flags flags =  CL_MEM_READ_WRITE );
    virtual ~Tensor();

    size_t getSize(int index) const;
    std::vector<int> const &getSizes() const;
    void setSizes(std::vector<int> const &sizes);
    std::vector<float> read() const;
    cl::Buffer getBuffer() const;
    cl::Buffer getSizesBuffer() const;
    size_t getOffset() const;
    size_t getNbElements() const;
    bool dataEquals(Tensor const &o, float tolerance = 0.00001) const;

    Tensor &add(float value);
    Tensor &sub(float value);
    Tensor &mul(float value);
    Tensor &div(float value);
    Tensor &add(Tensor const &o);
    Tensor &sub(Tensor const &o);
    Tensor &mul(Tensor const &o);
    Tensor &div(Tensor const &o);
    Tensor &fill(float value);

  protected:
    size_t _offset;

  public:
    std::shared_ptr<Storage<float>> _storage;
    std::unique_ptr<Storage<int>> _sizesStorage;

  private:

    std::vector<int> _sizes;


  private:
    // Private copy constructor to avoid leaks
    Tensor(Tensor &o) { std::cerr << "Something called the copy constructor" << std::endl; }
    Tensor(Tensor const &o) { std::cerr << "Something called the copy constructor" << std::endl; }
  };

  std::ostream &operator<<(std::ostream &s, Tensor const &t);
}

#endif
