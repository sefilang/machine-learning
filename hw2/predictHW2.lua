
require 'torch'
require 'image'
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'

function predict(model_name)

local trainset = torch.load('cifar.torch/cifar10-train.t7')
local testset = torch.load('cifar.torch/cifar10-test.t7')

local trainData = trainset.data:float() -- convert the data from a ByteTensor to a float Tensor.
local testData = testset.data:float()
local testLabels = testset.label:float():add(1)


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

local classes = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'}


model = torch.load(model_name)
model:evaluate()
model:cuda()

local confusion = optim.ConfusionMatrix(classes)

--local confusion = optim.ConfusionMatrix(torch.range(0,9):totable())

--data=testData:cuda()
--labels=testLabels:cuda()

local batchSize = 128
function forwardNet(data,labels)
    local confusion = optim.ConfusionMatrix(classes)
    local lossAcc = 0
    local numBatches = 0
    
    for i = 1, data:size(1) - batchSize, batchSize do
        numBatches = numBatches + 1
        local x = data:narrow(1, i, batchSize):cuda()
        local yt = labels:narrow(1, i, batchSize):cuda()
        local y = model:forward(x)
		
        confusion:batchAdd(y,yt)
   
    end
    
    confusion:updateValids()
    local avgError = 1 - confusion.totalValid
    
    return avgError
end
local _error = forwardNet(testData,testLabels)
     
    return  _error
end

a=predict('hw2.t7')
print (a)