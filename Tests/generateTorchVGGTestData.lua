require 'nn'
require 'loadcaffe'

local proto_file = "/home/wilmot_p/.models/VGG_ILSVRC_19_layers_deploy.prototxt"
local model_file = "/home/wilmot_p/.models/VGG_ILSVRC_19_layers.caffemodel"
local vgg = loadcaffe.load(proto_file, model_file, 'nn'):float()
while vgg:get(#vgg).name ~= 'relu5_1' do vgg:remove(#vgg) end
for i=1, #vgg do
   vgg:get(i).gradInput = nil
   vgg:get(i).gradBias = nil
   if torch.type(vgg:get(i)) == "nn.ReLU" then
      vgg:get(i).inplace = false
   end
end
torch.save('vgg.t7', vgg:float(), 'ascii')

local dilated = nn.Sequential()
local spacer = 1
for i=1, #vgg do
   local layer = vgg:get(i)
   local newLayer = layer
   if torch.type(layer) == "nn.SpatialConvolution" then
      dilated:add(nn.SpatialReflectionPadding(spacer, spacer, spacer, spacer))
      newLayer = nn.SpatialDilatedConvolution(layer.nInputPlane, layer.nOutputPlane, layer.kW, layer.kH, 1, 1, 0, 0, spacer, spacer)
      newLayer.weight:copy(layer.weight)
      newLayer.bias:copy(layer.bias)
   end
   if torch.type(layer) == "nn.SpatialMaxPooling" then
      spacer = spacer * 2
   else
      dilated:add(newLayer)
   end
end
torch.save('dilatedVGGReflectionPadding.t7', dilated:float(), 'ascii')

vgg:forward(input)
for i=1, #vgg do
   torch.save('vggLayer'..i..'Output.t7', vgg:get(i).output:float(), 'ascii')
end
