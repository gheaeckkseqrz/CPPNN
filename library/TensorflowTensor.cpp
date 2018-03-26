#include <cassert>
#include <string>
#include "TensorflowParser.h"
#include "TensorflowTensor.h"

namespace NN
{
  TensorflowTensor::TensorflowTensor()
  {
  }

  TensorflowTensor::TensorflowTensor(std::string const &s)
  {
    init(s);
  }

  void TensorflowTensor::init(std::string const &s)
  {
    std::string value = TensorflowParser::getInstance()->getFirstElementOfType("value", s);
    std::string tensor = TensorflowParser::getInstance()->getFirstElementOfType("tensor", value);
    if (tensor != "")
      {
	std::string tensor_shape = TensorflowParser::getInstance()->getFirstElementOfType("tensor_shape", tensor);
	std::list<std::pair<std::string, std::string>> dims = TensorflowParser::getInstance()->getAllElementsOfType("dim", tensor_shape);
	std::vector<int> sizes;
	int nbElem = 1;
	if (dims.size() > 0)
	  {
	    for (std::pair<std::string, std::string> dim : dims)
	      {
		std::string size = TensorflowParser::getInstance()->getAttribute("size", dim.second);
		if (size != "")
		  {
		    sizes.push_back(std::atoi(size.c_str()));
		    nbElem *= sizes.back();
		  }
	      }
	  }
	else
	  sizes.push_back(1);
	setSizes(sizes);

	std::vector<float> values(nbElem);
	std::string tensor_content = TensorflowParser::getInstance()->getAttribute("tensor_content", tensor);
	if (tensor_content != "")
	  {
	    int index = 0;
	    int i = 0;
	    while (index < tensor_content.size())
	      {
		values[i] = parseFloat(tensor_content, index);
		i++;
	      }
	  }
	else
	  {
	    std::string float_val = tensor_content = TensorflowParser::getInstance()->getAttribute("float_val", tensor);
	    if (float_val != "")
	      {
		for (int i = 0 ; i < values.size() ; ++i)
		  values[i] = std::atof(float_val.c_str());
	      }
	    std::string int_val = tensor_content = TensorflowParser::getInstance()->getAttribute("int_val", tensor);
	    if (int_val != "")
	      {
		for (int i = 0 ; i < values.size() ; ++i)
		  values[i] = std::atoi(int_val.c_str());
	      }
	  }
	_storage = std::make_shared<Storage<float>>(values);
      }
  }

  float TensorflowTensor::parseFloat(std::string const &s, int &index)
  {
    // Doing some byte array to float conversion.
    // Need to make sure a float is actually 4 bytes
    assert(sizeof(char) == 1 && sizeof(float) == 4);
    char b[4];
    for (int i = 0; i < 4; ++i)
      {
    	if (s[index] == '\\')
    	  {
    	    if (s[index + 1] < '0' || s[index + 1] > '9')
    	      {
    		if (s[index + 1] == 't')
    		  b[i] = '\t';
    		else if (s[index + 1] == 'n')
    		  b[i] = '\n';
    		else if (s[index + 1] == 'v')
    		  b[i] = '\v';
    		else if (s[index + 1] == 'f')
    		  b[i] = '\f';
    		else if (s[index + 1] == 'r')
    		  b[i] = '\r';
    		else if (s[index + 1] == 'a')
    		  b[i] = '\a';
    		else if (s[index + 1] == 'b')
    		  b[i] = '\b';
    		else
    		  b[i] = s[index + 1];
    		index += 2;
    	      }
    	    else
    	      {
    		b[i] = (char)std::stoi(s.substr(index + 1, 3), nullptr, 8);
    		index += 4;
    	      }
    	  }
    	else
    	  {
    	    b[i] = s[index];
    	    index += 1;
    	  }
      }
    float f = *((float*)b);
    return f;
  }
}
