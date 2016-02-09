----------------------------------------------------------------------
-- This script uses the saved model to output a csv file with predictions
-- on the test data.
--
-- Team DeepPurple
----------------------------------------------------------------------

require 'torch'   -- torch
require 'image'
require 'optim'
require 'nn'
require 'csvigo'
require 'xlua'
require 'lfs' --comment out later
path = "."
--path = "/Users/shivamverma/Desktop/NYU/Spring_16/Deep_Learning/assignments/ds-ga-1008-a1/results/"
----------------------------------------------------------------------
-- parse command line arguments
if not opt then
   print '==> processing options'
   cmd = torch.CmdLine()
   cmd:text()
   cmd:text('MNIST Dataset Preprocessing')
   cmd:text()
   cmd:text('Options:')
   cmd:option('-size', 'small', 'how many samples do we load: small | full')
   cmd:option('-visualize', true, 'visualize input data and weights during training')
   cmd:option('-type', 'double', 'type: double | float | cuda')
   cmd:option('-loss', 'nll', 'type of loss function to minimize: nll | mse | margin')
   cmd:text()
   opt = cmd:parse(arg or {})
end
----------------------------------------------------------------------
function getPrediction(pred)
    local y, ind = torch.max(pred,1)
    return ind[1]
end

if paths.filep(path .. "model.net") then
    local model = torch.load( path .. "model.net")
    model:evaluate()
    print(model)
    dofile '1_data.lua' --this is to obtain normalized testData

    predTable = {}
    predTable[1] = {"Id","Prediction"}

    print('==> testing on test set:')
    print(testData:size())
    for t = 1,testData:size() do
      -- disp progress
        xlua.progress(t, testData:size())

      -- get new sample
        local input = testData.data[t]
        if opt.type == 'double' then input = input:double()
        elseif opt.type == 'cuda' then input = input:cuda() end
        local target = testData.labels[t]

        local pred = model:forward(input)
        local plabel = getPrediction(pred)
        predTable[t+1] = {t,plabel}

   end
   --print(predTable)


   -- print(confusion)
   --    -- update log/plot
   -- testLogger:add{['% mean class accuracy (test set)'] = confusion.totalValid * 100}
   -- if opt.plot then
   --    testLogger:style{['% mean class accuracy (test set)'] = '-'}
   --    testLogger:plot()
   -- end

   -- print(predTable)

    print('==> Generating predictions.csv')
    f = io.open('predictions.csv', 'w')
    for i = 1, #predTable do
       f:write(predTable[i][1] .. ',' .. predTable[i][2] .. '\n')  -- You should know that \n brings newline and .. concats stuff
    end
    f:close()
    print('==> predictions.csv is created!')

else
    print "Error: model.net not present in current directory."

end
