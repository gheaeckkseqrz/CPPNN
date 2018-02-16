from os import listdir
from os.path import isfile, join
from parse import *

def getUtilitiesFunctions():
    with open("Tensor.h") as f:
        return f.read()

def expandTensorParameter(tensorName):
    s = "__global float *" + tensorName + "_data, "
    s += "__global int * " + tensorName + "_dims, "
    s += "int " + tensorName + "_nbDims, "
    s += "int " + tensorName + "_offset, "
    return s

def expandTensorSetArg(tensorName, i):
    s  = "kernel.setArg(" + str(i) +   ", " + tensorName + ".getBuffer());\n"
    s += "kernel.setArg(" + str(i+1) + ", " + tensorName + ".getSizesBuffer());\n"
    s += "kernel.setArg(" + str(i+2) + ", " + tensorName + ".getSizes().size());\n"
    s += "kernel.setArg(" + str(i+3) + ", " + tensorName+  ".getOffset());\n"
    return s

def getParameterTypeAndName(parameter):
    s = parameter.split(' ')
    for i in range(len(s)):
        s[i] = filter(lambda a: a!= '\n' and a != '\t', s[i])
    s = filter(lambda a: a!= '', s)
    if (s[0] == "__global"):
        s.pop(0)
    return s[0], s[1]

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
            parameterType, parameterName = getParameterTypeAndName(p)
            if parameterType == "Tensor":
                kernel += expandTensorParameter(parameterName)
            else:
                kernel += parameterType + " " +  parameterName + ", "
        kernel = kernel[:-2] # Remove last coma
        kernel += ")"
        kernel += body

        function  = "void "
        function += "OpenCLFuncs::" + functionName + "("
        for p in parameters.split(','):
            parameterType, parameterName = getParameterTypeAndName(p)
            if parameterType == "Tensor":
                parameterType = "Tensor const &"
            function += parameterType + " " +  parameterName + ", "
        function += "size_t nbThread)\n"
        function += "{\n"
        function += "std::string kernelSource = R\"RAW(" + kernel + ")RAW\"; \n"
        function += "cl::Program program = OpenCL::getInstance()->buildProgramFromSource(kernelSource);\n"
        function += "cl::Kernel kernel = cl::Kernel(program, \"" + functionName + "\");\n"
        i = 0
        for p in parameters.split(','):
            parameterType, parameterName = getParameterTypeAndName(p)
            if parameterType == "Tensor":
                function += expandTensorSetArg(parameterName, i)
                i += 4
            else:
                function += "kernel.setArg(" + str(i) + ", " + parameterName + ");\n"
                i += 1
        function += "clock_t begin = clock();\n"
        function += "std::cout << \"Running kernel [" + functionName + "] with \" << nbThread << \" threads - \";\n"
        function += "OpenCL::getInstance()->runKernel(kernel, nbThread);\n"
        function += "clock_t end = clock();\n"
        function += "std::cout << double(end - begin) / CLOCKS_PER_SEC << \" sec\" << std::endl;\n"
        function += "}\n"

        prototype = "void " + functionName + "("
        for p in parameters.split(','):
            parameterType, parameterName = getParameterTypeAndName(p)
            if parameterType == "Tensor":
                parameterType = "Tensor const &"
            prototype += parameterType + " " + parameterName + ", "
        prototype += "size_t nbThread);\n"

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
