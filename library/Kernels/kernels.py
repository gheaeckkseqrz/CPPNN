from os import listdir
from os.path import isfile, join
from parse import *

def toCLType(typeName):
    if typeName == "int":
        return "(cl_int)"
    if typeName == "float":
        return "(cl_float)"
    return ""

def getUtilitiesFunctions():
    with open("Tensor.h") as f:
        return f.read()

def expandTensorParameter(tensorName, addressSpace):
    addressSpace = "__global" if addressSpace == "" else addressSpace
    s = addressSpace + " float *" + tensorName + "_data, "
    s += "__global int * " + tensorName + "_dims, "
    s += "int " + tensorName + "_nbDims, "
    s += "int " + tensorName + "_offset, "
    return s

def expandTensorSetArg(tensorName, i):
    s  = "kernel.setArg(" + str(i) +   ", " + tensorName + ".getBuffer());\n"
    s += "kernel.setArg(" + str(i+1) + ", " + tensorName + ".getSizesBuffer());\n"
    s += "kernel.setArg(" + str(i+2) + ", (cl_int)" + tensorName + ".getSizes().size());\n"
    s += "kernel.setArg(" + str(i+3) + ", (cl_int)" + tensorName+  ".getOffset());\n"
    return s

def getCPPParameterTypeAndName(parameter):
    s = parameter.split(' ')
    for i in range(len(s)):
        s[i] = filter(lambda a: a!= '\n' and a != '\t', s[i])
    s = filter(lambda a: a!= '', s)
    if (s[0] == "__global" or s[0] == "__constant"):
        s.pop(0)
    if (s[0] == "__local"):
        s.pop(0)
        s[0] = "int"
        if s[1][0] == '*':
            s[1] = s[1][1:]
    return s[0], s[1]

def getCLParameterSpaceAndTypeAndName(parameter):
    addressSpace = ""
    s = parameter.split(' ')
    for i in range(len(s)):
        s[i] = filter(lambda a: a!= '\n' and a != '\t', s[i])
    s = filter(lambda a: a!= '', s)
    if (s[0] == "__global" or s[0] == "__local" or s[0] == "__constant"):
        addressSpace = s[0]
        s.pop(0)
    return addressSpace, s[0], s[1]

def parsePrototype(content):
    res = parse("kernel {return} {name}({parameters}){body}", content)
    if res == None:
        print("Can't parse from:\n\n")
        print("=============================================")
        print(content)
        print("=============================================")
    return res['return'], res['name'], res['parameters'], res['body']

def generateFunction(filePath):
    with open(filePath) as f:
        content = f.read()
        returnType, functionName, parameters, body = parsePrototype(content)

        kernel = getUtilitiesFunctions()
        kernel += "kernel "
        kernel += returnType + " "
        kernel += functionName + "("
        for p in parameters.split(','):
            parameterSpace, parameterType, parameterName = getCLParameterSpaceAndTypeAndName(p)
            if parameterType == "Tensor":
                kernel += expandTensorParameter(parameterName, parameterSpace)
            else:
                kernel += parameterSpace + " " + parameterType + " " +  parameterName + ", "
        kernel = kernel[:-2] # Remove last coma
        kernel += ")"
        kernel += body

        function  = "void "
        function += "OpenCLFuncs::" + functionName + "("
        for p in parameters.split(','):
            parameterType, parameterName = getCPPParameterTypeAndName(p)
            if parameterType == "Tensor":
                parameterType = "Tensor const &"
            function += parameterType + " " +  parameterName + ", "
        function += "size_t nbThread, size_t groupSize)\n"
        function += "{\n"
        function += "std::string kernelSource = R\"RAW(" + kernel + ")RAW\"; \n"
        function += "if (_compiledKernels.count(\"" + functionName + "\") == 0)\n"
        function += "_compiledKernels[\"" + functionName + "\"] = OpenCL::getInstance()->buildProgramFromSource(kernelSource);\n"
        function += "cl::Kernel kernel = cl::Kernel(_compiledKernels[\"" + functionName + "\"], \"" + functionName + "\");\n"
        i = 0
        for p in parameters.split(','):
            CPPparameterType, CPPparameterName = getCPPParameterTypeAndName(p)
            CLparameterSpace, CLparameterType, CLparameterName = getCLParameterSpaceAndTypeAndName(p)
            if CPPparameterType == "Tensor":
                function += expandTensorSetArg(CPPparameterName, i)
                i += 4
            elif CLparameterSpace == "__local":
                function += "kernel.setArg(" + str(i) + ", cl::Local(" + CPPparameterName + " * sizeof(cl_float)));\n"
                function += "std::cout << \"Allocating local memory of size \" << " + CPPparameterName + " << std::endl;\n"
                i += 1
            else:
                function += "kernel.setArg(" + str(i) + ", " + toCLType(CPPparameterType) + CPPparameterName + ");\n"
                i += 1
        function += "clock_t begin = clock();\n"
        function += "std::cout << \"Running kernel [" + functionName + "] with \" << nbThread << \" threads - (groupSize : \" << (int)groupSize << \") \";\n"
        function += "OpenCL::getInstance()->runKernel(kernel, nbThread, groupSize);\n"
        function += "clock_t end = clock();\n"
        function += "std::cout << double(end - begin) / CLOCKS_PER_SEC << \" sec\" << std::endl;\n"
        function += "}\n"

        prototype = "void " + functionName + "("
        for p in parameters.split(','):
            parameterType, parameterName = getCPPParameterTypeAndName(p)
            if parameterType == "Tensor":
                parameterType = "Tensor const &"
            prototype += parameterType + " " + parameterName + ", "
        prototype += "size_t nbThread, size_t groupSize = -1);\n"

        return prototype, function

kernelFolder = "."
fileList = [f for f in listdir(kernelFolder) if isfile(join(kernelFolder, f)) and f.endswith(".cl")]

functions = ""
prototypes = ""
for f in fileList:
    print("Processing " + f)
    prototype, function = generateFunction(f)
    functions += function + "\n"
    prototypes += prototype + "\n"

with open("OpenCLFuncs.h.in") as f:
    content = f.read()
    content = content.replace("PROTOTYPES;", prototypes)
    with open("../OpenCLFuncs.h", "w+") as f2:
        f2.write(content)

with open("OpenCLFuncs.cpp.in") as f:
    content = f.read()
    content = content.replace("FUNCTIONS;", functions)
    with open("../OpenCLFuncs.cpp", "w+") as f2:
        f2.write(content)
