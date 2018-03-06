#include <fstream>
#include "OpenCL.h"

OpenCL *OpenCL::_instance = nullptr;

OpenCL::OpenCL()
{
  _device = getOpenCLDevice();
  _context = cl::Context({_device});
  _queue = cl::CommandQueue(_context, _device);
}

OpenCL *OpenCL::getInstance()
{
  if (_instance == nullptr)
    _instance = new OpenCL();
  return _instance;
}

OpenCL::~OpenCL()
{
  _queue.finish();
  _instance = nullptr;
}

cl::Device OpenCL::getOpenCLDevice()
{
  std::vector<cl::Platform> all_platforms;
  cl::Platform::get(&all_platforms);
  if(all_platforms.size() == 0)
    {
      std::cout<<" No platforms found. Check OpenCL installation!\n";
      exit(1);
    }
  cl::Platform default_platform = all_platforms[0];
  std::cout << "Using platform: " << default_platform.getInfo<CL_PLATFORM_NAME>() << std::endl;
  std::vector<cl::Device> all_devices;
  default_platform.getDevices(CL_DEVICE_TYPE_ALL, &all_devices);
  if(all_devices.size() == 0)
    {
      std::cout<<" No devices found. Check OpenCL installation!\n";
      exit(1);
    }
  std::cout<< "Using device: "<< all_devices[0].getInfo<CL_DEVICE_NAME>() << std::endl;
  return all_devices[0];
}

std::vector<char> OpenCL::getFileContent(std::string const &path)
{
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (file.is_open())
    {
      std::streamsize size = file.tellg();
      file.seekg(0, std::ios::beg);
      std::vector<char> buffer(size);
      if (file.read(buffer.data(), size))
	return buffer;
    }
  std::cerr << "Could not open file [" << path << "]" << std::endl;
  exit(-1);
}

cl::Program OpenCL::buildProgramFromSource(std::string const &source)
{
  std::vector<char> v(source.begin(), source.end());
  return buildProgramFromSource(v);
}

cl::Program OpenCL::buildProgramFromSource(std::vector<char> const &source)
{
  cl::Program::Sources sources;
  sources.push_back({source.data(), source.size()});

  cl::Program program(_context, sources);
  if(program.build({_device}) != CL_SUCCESS)
    {
      std::cerr << "Error building: "<<program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(_device) << std::endl;
      exit(1);
  }
  return program;
}

cl::Program OpenCL::buildProgramFromFile(std::string const &path)
{
  std::vector<char> source = getFileContent(path);
  return buildProgramFromSource(source);
}

void OpenCL::runKernel(cl::Kernel &kernel, unsigned int workItems, unsigned int groupSize)
{

  size_t result;

  _device.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &result);
  // std::cout << "Max LocalMemory size for kernel is " << result << std::endl;

  kernel.getWorkGroupInfo(_device, CL_KERNEL_WORK_GROUP_SIZE, &result);
  // std::cout << "Max WorkGroup size for kernel is " << result << std::endl;

  cl::NDRange global = cl::NDRange(workItems);
  cl::NDRange local = (groupSize == -1) ?  cl::NullRange : cl::NDRange(groupSize);
  int ret = _queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local);
  _queue.finish();
  if (ret != CL_SUCCESS)
    std::cerr << "runKernel returned error " << ret << std::endl;
}
