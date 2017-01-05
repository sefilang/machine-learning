
require 'torch'
require 'image'
require 'nn'
require 'cunn'
require 'cudnn'

local function Tower(layers)
  local tower = nn.Sequential()
  for i=1,#layers do
    tower:add(layers[i])
  end
  return tower
end

local function FilterConcat(towers)
  local concat = nn.DepthConcat(2)
  for i=1,#towers do
    concat:add(towers[i])
  end
  return concat
end

function saveTensorAsGrid(tensor,fileName) 
	local padding = 1
	local grid = image.toDisplayTensor(tensor/255.0, padding)
	image.save(fileName,grid)
end

local trainset = torch.load('cifar.torch/cifar10-train.t7')
local testset = torch.load('cifar.torch/cifar10-test.t7')

local classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}

local trainData = trainset.data:float() -- convert the data from a ByteTensor to a float Tensor.
local trainLabels = trainset.label:float():add(1)
local testData = testset.data:float()
local testLabels = testset.label:float():add(1)

--  ****************************************************************
-- Training a ConvNet on Cifar10
--  ****************************************************************

-- Load and normalize data:


local mean = {}  -- store the mean, to normalize the test set in the future
local stdv  = {} -- store the standard-deviation for the future
for i=1,3 do -- over each image channel
    mean[i] = trainData[{ {}, {i}, {}, {}  }]:mean() -- mean estimation
    trainData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction
    
    stdv[i] = trainData[{ {}, {i}, {}, {}  }]:std() -- std estimation
    trainData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end

-- Normalize test set using same values

for i=1,3 do -- over each image channel
    testData[{ {}, {i}, {}, {}  }]:add(-mean[i]) -- mean subtraction    
    testData[{ {}, {i}, {}, {}  }]:div(stdv[i]) -- std scaling
end



--  ****************************************************************
--  Define neural network
--  ****************************************************************

local model = nn.Sequential()
model:add(cudnn.SpatialConvolution(3, 60, 5, 5,1,1,2,2)) -- 3 input image channel, 32 output channels, 5x5 convolution kernel
model:add(nn.SpatialBatchNormalization(60,1e-3))
model:add(cudnn.ReLU(true))
                
model:add(cudnn.SpatialConvolution(60,42,1,1))
model:add(nn.SpatialBatchNormalization(42,1e-3))
model:add(cudnn.ReLU(true))

model:add(cudnn.SpatialConvolution(42,30,3,3,1,1,1,1))
model:add(nn.SpatialBatchNormalization(30,1e-3))
model:add(cudnn.ReLU(true))

model:add(nn.SpatialMaxPooling(3,3,2,2):ceil())
model:add(nn.Dropout(0.3))

model:add(FilterConcat(
{
Tower({cudnn.SpatialConvolution(30, 32, 1, 1), -- 3 input image channel, 32 output channels, 5x5 convolution kernel
nn.SpatialBatchNormalization(32,1e-3),
cudnn.ReLU(true)}),
Tower({cudnn.SpatialConvolution(30, 32, 3, 3,1,1,1,1), -- 3 input image channel, 32 output channels, 5x5 convolution kernel
nn.SpatialBatchNormalization(32,1e-3),
cudnn.ReLU(true)})

}
))  
model:add(cudnn.SpatialConvolution(64, 46, 1, 1,1,1)) -- 3 input image channel, 32 output channels, 5x5 convolution kernel
model:add(nn.SpatialBatchNormalization(46,1e-3))
model:add(cudnn.ReLU(true))

model:add(nn.SpatialAveragePooling(3,3,2,2):ceil())
model:add(nn.Dropout(0.3))

model:add(cudnn.SpatialConvolution(46, 26, 3, 3,1,1,1,1)) -- 3 input image channel, 32 output channels, 5x5 convolution kernel
model:add(nn.SpatialBatchNormalization(26,1e-3))
model:add(cudnn.ReLU(true))

model:add(cudnn.SpatialConvolution(26, 30, 3, 3,1,1,1,1)) -- 3 input image channel, 32 output channels, 5x5 convolution kernel
model:add(nn.SpatialBatchNormalization(30,1e-3))
model:add(cudnn.ReLU(true))


model:add(cudnn.SpatialConvolution(30, 10, 1, 1,1,1)) -- 3 input image channel, 32 output channels, 5x5 convolution kernel
model:add(nn.SpatialBatchNormalization(10,1e-3))
model:add(cudnn.ReLU(true))

model:add(nn.SpatialMaxPooling(8,8,1,1):ceil())

model:add(nn.View(10))
model:add(nn.LogSoftMax())                     -- converts the output to a log-probability. Useful for classificati

model:cuda()
criterion = nn.ClassNLLCriterion():cuda()

for k,v in pairs(model:findModules(('%s.SpatialConvolution'):format(backend_name))) do

  v.weight:normal(0,0.05)

  v.bias:zero()

end

w, dE_dw = model:getParameters()

function shuffle(data,ydata) --shuffle data function
    local RandOrder = torch.randperm(data:size(1)):long()
    return data:index(1,RandOrder), ydata:index(1,RandOrder)
end

--  ****************************************************************
--  Training the network
--  ****************************************************************
require 'optim'

local batchSize = 128
local optimState = {}
function forwardNet(data,labels, train)
    local confusion = optim.ConfusionMatrix(classes)
    local lossAcc = 0
    local numBatches = 0
    if train then
        --set network into training mode
        model:training()
    else
        model:evaluate() -- turn of drop-out
    end
    for i = 1, data:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        local x = data:narrow(1, i, batchSize):cuda()
        local yt = labels:narrow(1, i, batchSize):cuda()
        local y = model:forward(x)
        local err = criterion:forward(y, yt)
        lossAcc = lossAcc + err
        confusion:batchAdd(y,yt)
        
        if train then
            function feval()
                model:zeroGradParameters() --zero grads
                local dE_dy = criterion:backward(y,yt)
                model:backward(x, dE_dy) -- backpropagation
            
                return err, dE_dw
            end
        
            optim.adam(feval, w, optimState)
        end
    end
    
    confusion:updateValids()
    local avgLoss = lossAcc / numBatches
    local avgError = 1 - confusion.totalValid
    
    return avgLoss, avgError, tostring(confusion)
end


require 'image'
require 'nn'


local function randomcrop(im , pad, randomcrop_type)
   if randomcrop_type == 'reflection' then
      -- Each feature map of a given input is padded with the replication of the input boundary
      module = nn.SpatialReflectionPadding(pad,pad,pad,pad):float() 
   elseif randomcrop_type == 'zero' then
      -- Each feature map of a given input is padded with specified number of zeros.
	  -- If padding values are negative, then input is cropped.
      module = nn.SpatialZeroPadding(pad,pad,pad,pad):float()
   end
	
   local padded = module:forward(im:float())
   local x = torch.random(1,pad*2 + 1)
   local y = torch.random(1,pad*2 + 1)

   return padded:narrow(3,x,im:size(3)):narrow(2,y,im:size(2))
end

 -- data augmentation module
--{'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}
 local function updateOutput(input,labels)
      local permutation = torch.randperm(input:size(1))
      for i=1,input:size(1) do
        if 0 == permutation[i] % 3  then input[i]=image.hflip(input[i]) end -- need to define f
		--if 1 == permutation[i] % 9  then input[i]=randomcrop(input[i],8,'reflection') end 
		--if 2 == permutation[i] % 9  then input[i]=randomcrop(input[i],6,'zero') end 
		
		--if (3 == permutation[i] % 6  and labels[i] ~= 'airplane' and labels[i] ~= 'automobile' and labels[i] ~= 'ship' and labels[i] ~= 'truck') then input[i]=image.rotate(input[i],0.38) end 
		--if (4 == permutation[i] % 6  and labels[i] ~= 'airplane' and labels[i] ~= 'automobile' and labels[i] ~= 'ship' and labels[i] ~= 'truck') then input[i]=image.rotate(input[i],-0.38) end
      end
	
    return input
  end

---------------------------------------------------------------------

epochs =600
trainLoss = torch.Tensor(epochs)
testLoss = torch.Tensor(epochs)
trainError = torch.Tensor(epochs)
testError = torch.Tensor(epochs)


timer = torch.Timer()

for e = 1, epochs do
    trainData, trainLabels = shuffle(trainData, trainLabels) --shuffle training data
    trainLoss[e], trainError[e] = forwardNet(updateOutput(trainData,trainLabels), trainLabels, true)
    testLoss[e], testError[e], confusion = forwardNet(testData, testLabels, false)
end

