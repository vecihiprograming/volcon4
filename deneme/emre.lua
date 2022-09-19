require 'dpnn'
require 'rnn'
require 'nngraph'
require 'optim'
require 'image'

require 'VRMSEReward'
require 'SpatialGlimpse_inverse'


util = paths.dofile('util.lua')
-- nngraph.setDebug(true)
opt = lapp[[
   -b,--batchSize             (default 1)         batch size
   -r,--lr                    (default 0.0002)    learning rate

   --dataset                  (default 'folder')  imagenet / lsun / folder
   --nThreads                 (default 1)         # of data loading threads to use

   --beta1                    (default 0.5)       momentum term of adam
   --ntrain                   (default math.huge) #  of examples per epoch. math.huge for full dataset
   --display                  (default 0)         display samples while training. 0 = false
   --display_id               (default 10)        display window id.
   --gpu                      (default 0)         gpu = 0 is CPU mode. gpu=X is GPU mode on GPU X
   --GAN_loss_after_epoch     (default 5)
   --name                     (default 'fullmodel')
   --checkpoints_name         (default '')        name of checkpoints for load
   --checkpoints_epoch        (default 0)         epoch of checkpoints for load
   --epoch                    (default 1)         save checkpoints every N epoch
   --nc                       (default 3)         number of input image channels (RGB/Grey)

   --niter                    (default 250)  maximum number of iterations

   --rewardScale              (default 1)     scale of positive reward (negative is 0)
   --rewardAreaScale          (default 4)     scale of aree reward
   --locatorStd               (default 0.11)  stdev of gaussian location sampler (between 0 and 1) (low values may cause NaNs)')

   --glimpseHiddenSize        (default 128)   size of glimpse hidden layer')
   --glimpsePatchSize         (default '60,45')     size of glimpse patch at highest res (height = width)')
   --glimpseScale             (default 1)     scale of successive patches w.r.t. original input image')
   --glimpseDepth             (default 1)     number of concatenated downscaled patches')
   --locatorHiddenSize        (default 128)   size of locator hidden layer')
   --imageHiddenSize          (default 512)   size of hidden layer combining glimpse and locator hiddens')
   --wholeImageHiddenSize     (default 256)   size of full image hidden size

   --pertrain_SR_loss         (default 2)     SR loss before training action
   --residual                 (default 1)     whether learn residual in each step
   --rho                      (default 25)    back-propagate through time (BPTT) for rho time-steps
   --hiddenSize               (default 512)   number of hidden units used in Simple RNN.
   --FastLSTM                 (default 1)     use LSTM instead of linear layer
   --BN                                       whether use BatchNormalization
   --save_im                                  whether save image on test
]]

-- one-line argument parser. parses enviroment variables to override the defaults
for k,v in pairs(opt) do opt[k] = tonumber(os.getenv(k)) or os.getenv(k) or opt[k] end
print(opt)
if opt.ntrain == 'math.huge' then opt.ntrain = math.huge end
-- image size for dataloder, high-resolution and low-resolution
opt.loadSize = {128, 128}
opt.highResSize = {128, 128}
opt.lowResSize = {16, 16}
-- local patch size
local PatchSize = {}
PatchSize[1], PatchSize[2] = opt.glimpsePatchSize:match("([^,]+),([^,]+)")
opt.glimpsePatchSize = {}
opt.glimpsePatchSize[1] = tonumber(PatchSize[1])
opt.glimpsePatchSize[2] = tonumber(PatchSize[2])
opt.glimpseArea = opt.glimpsePatchSize[1]*opt.glimpsePatchSize[2]
if opt.glimpseArea == opt.highResSize[1]*opt.highResSize[2] then
  opt.unitPixels = (opt.highResSize[2] - opt.glimpsePatchSize[2]) / 2
else
  opt.unitPixels = opt.highResSize[2] / 2
end
if opt.display == 0 then opt.display = false end -- lapp argparser cannot handel bool value 

opt.manualSeed = 123 --torch.random(1, 10000) -- fix seed
print("Random Seed: " .. opt.manualSeed)
torch.manualSeed(opt.manualSeed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- create train data loader
local DataLoader = paths.dofile('data/data.lua')
opt.data = './denemelfw128/train/'
local data = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("DatasetTrain: " .. opt.dataset, " Size: ", data:size())
opt.data = './denemelfw128/test/'
local dataTest = DataLoader.new(opt.nThreads, opt.dataset, opt)
print("DatasetTest: " .. opt.dataset, " Size: ", dataTest:size())
----------------------------------------------------------------------------
local function weights_init(m)
    local name = torch.type(m)
    if name:find('Convolution') or name:find('Linear') then
      if m.weight then m.weight:normal(0.0, 0.02) end
      if m.bias then m.bias:normal(0.0, 0.02) end
    elseif name:find('BatchNormalization') then
      if m.weight then m.weight:normal(1.0, 0.02) end
      if m.bias then m.bias:normal(0.0, 0.02) end
    end
end

local nc = opt.nc
local rho = opt.rho
local lowResSize = opt.lowResSize
local highResSize = opt.highResSize

local SpatialBatchNormalization

if opt.BN then SpatialBatchNormalization = nn.SpatialBatchNormalization
else SpatialBatchNormalization = nn.Identity end
local SpatialConvolution = nn.SpatialConvolution

if opt.checkpoints_epoch and opt.checkpoints_epoch > 0 then
  nngraph.annotateNodes()
  print('Loading.. checkpoints_final/' .. opt.checkpoints_name .. '_' .. opt.checkpoints_epoch .. '_RNN.t7')
  model = torch.load('checkpoints_final/' .. opt.checkpoints_name .. '_' .. opt.checkpoints_epoch .. '_RNN.t7')
else
  ----------------------- locator net -----------------------
  -- Encode the (x,y) -- coordinate of last attended patch
  local locationSensor = nn.Sequential()
  locationSensor:add(nn.Linear(2, opt.locatorHiddenSize))
  locationSensor:add(nn.BatchNormalization(opt.locatorHiddenSize)):add(nn.ReLU(true))
  print("locationSensor----->",locationSensor)

  -- Encode the low-resolution input image
  local imageSensor = nn.Sequential()
  imageSensor:add(nn.View(-1):setNumInputDims(3)) -- forwardda yap
  imageSensor:add(nn.Linear(nc*highResSize[1]*highResSize[2],opt.wholeImageHiddenSize))
  imageSensor:add(nn.BatchNormalization(opt.wholeImageHiddenSize)):add(nn.ReLU(true))
  print("imageSensor----->",imageSensor) 
  

   -- Encode the enhanced image in last step
  local imageErrSensor = nn.Sequential()
  imageErrSensor:add(nn.View(-1):setNumInputDims(3)) -- forwardda yap
  imageErrSensor:add(nn.Linear(nc*highResSize[1]*highResSize[2],opt.wholeImageHiddenSize))
  imageErrSensor:add(nn.BatchNormalization(opt.wholeImageHiddenSize)):add(nn.ReLU(true))
  print("imageErrSensor----->",imageErrSensor)  
  
  -- rnn input
  glimpse = nn.Sequential()
  glimpse:add(nn.ParallelTable():add(locationSensor):add(imageErrSensor):add(imageSensor))-- forwardda yap
  glimpse:add(nn.JoinTable(1,1))-- forwardda yap
  glimpse:add(nn.Linear(opt.wholeImageHiddenSize+opt.locatorHiddenSize+opt.wholeImageHiddenSize, opt.imageHiddenSize))
  glimpse:add(nn.BatchNormalization(opt.imageHiddenSize)):add(nn.ReLU(true))
  glimpse:add(nn.Linear(opt.imageHiddenSize, opt.hiddenSize))
  glimpse:add(nn.BatchNormalization(opt.hiddenSize)):add(nn.ReLU(true))
  print  ("glimpse----->",glimpse)
  -- rnn recurrent cell
  recurrent = nn.GRU(opt.hiddenSize, opt.hiddenSize)
  print("recurrent----->",recurrent)  
  -- recurrent neural network
  local rnn = nn.Recurrent(opt.hiddenSize, glimpse, recurrent, nn.ReLU(true), 99999)
  print("rnn----->",rnn)    
  
  -- output the coordinate of attended patch
  local locator = nn.Sequential()
  locator:add(nn.Linear(opt.hiddenSize, 2))
  locator:add(nn.Tanh()) -- bounds mean between -1 and 1
  locator:add(nn.ReinforceNormal(2*opt.locatorStd, opt.stochastic)) -- sample from normal, uses REINFORCE learning rule
  locator:add(nn.HardTanh()) -- bounds sample between -1 and 1, while reinforce recieve no gradInput
  locator:add(nn.MulConstant(opt.unitPixels*2/highResSize[2]))
  print("locator----->",locator)
 
  ----------------------- SR net -----------------------
  -- globally encode the attended patch
  local SR_patch_fc = nn.Sequential()
  SR_patch_fc:add(nn.JoinTable(1,3))
  SR_patch_fc:add(nn.View(-1):setNumInputDims(3))
  SR_patch_fc:add(nn.Linear(nc*2*opt.glimpsePatchSize[1]*opt.glimpsePatchSize[2],256)):add(nn.ReLU(true))
  SR_patch_fc:add(nn.Linear(256,256)):add(nn.ReLU(true))
  SR_patch_fc:add(nn.Linear(256,nc*opt.glimpsePatchSize[1]*opt.glimpsePatchSize[2])):add(nn.ReLU(true))
  SR_patch_fc:add(nn.View(nc,opt.glimpsePatchSize[1],opt.glimpsePatchSize[2]):setNumInputDims(1))
  -- globally encode the image
  local SR_img_fc = nn.Sequential()
  SR_img_fc:add(nn.JoinTable(1,3))
  SR_img_fc:add(nn.View(-1):setNumInputDims(3))
  SR_img_fc:add(nn.Linear(nc*2*highResSize[1]*highResSize[2],256)):add(nn.ReLU(true))
  SR_img_fc:add(nn.Linear(256,256)):add(nn.ReLU(true))
  SR_img_fc:add(nn.Linear(256,nc*opt.glimpsePatchSize[1]*opt.glimpsePatchSize[2])):add(nn.ReLU(true))
  SR_img_fc:add(nn.View(nc,opt.glimpsePatchSize[1],opt.glimpsePatchSize[2]):setNumInputDims(1))
  -- transform the hidden of RNN
  local SR_fc = nn.Sequential()
  SR_fc:add(nn.Linear(opt.hiddenSize,nc*opt.glimpsePatchSize[1]*opt.glimpsePatchSize[2])):add(nn.ReLU(true))
  SR_fc:add(nn.View(nc,opt.glimpsePatchSize[1],opt.glimpsePatchSize[2]):setNumInputDims(1))
  -- fully-convolution network for SR
  local SRnet = nn.Sequential()
  SRnet:add(nn.JoinTable(1,3))
  SRnet:add(SpatialConvolution(nc*5, 16, 5, 5, 1, 1, 2, 2))
  SRnet:add(SpatialBatchNormalization(16)):add(nn.ReLU(true))
  SRnet:add(SpatialConvolution(16, 32, 7, 7, 1, 1, 3, 3))
  SRnet:add(SpatialBatchNormalization(32)):add(nn.ReLU(true))
  SRnet:add(SpatialConvolution(32, 64, 7, 7, 1, 1, 3, 3))
  SRnet:add(SpatialBatchNormalization(64)):add(nn.ReLU(true))
  SRnet:add(SpatialConvolution(64, 64, 7, 7, 1, 1, 3, 3))
  SRnet:add(SpatialBatchNormalization(64)):add(nn.ReLU(true))
  SRnet:add(SpatialConvolution(64, 64, 7, 7, 1, 1, 3, 3))
  SRnet:add(SpatialBatchNormalization(64)):add(nn.ReLU(true))
  SRnet:add(SpatialConvolution(64, 32, 7, 7, 1, 1, 3, 3))
  SRnet:add(SpatialBatchNormalization(32)):add(nn.ReLU(true))
  SRnet:add(SpatialConvolution(32, 16, 5, 5, 1, 1, 2, 2))
  SRnet:add(SpatialBatchNormalization(16)):add(nn.ReLU(true))
  SRnet:add(SpatialConvolution(16, nc, 5, 5, 1, 1, 2, 2))

  -- nngraph build model
  -- input: {loc_prev, image_pre, image}
  -- output: {loc, image_next}
  local loc_prev = nn.Identity()()
  local image_pre = nn.Identity()()
  local image = nn.Identity()()
  local visited_map_pre = nn.Identity()() -- used for record the attened area
  local onesTensor = nn.Identity()()

  local h = rnn({loc_prev,image_pre,image})
  local loc = locator(h)
  local visited_map = nn.SpatialGlimpse_inverse(opt.glimpsePatchSize)({visited_map_pre, onesTensor, loc})
  local patch = nn.SpatialGlimpse(opt.glimpsePatchSize, opt.glimpseDepth, opt.glimpseScale)({image, loc})
  local patch_pre = nn.SpatialGlimpse(opt.glimpsePatchSize, opt.glimpseDepth, opt.glimpseScale)({image_pre, loc})
  local SR_patch_fc_o = SR_patch_fc({patch, patch_pre})
  local SR_img_fc_o = SR_img_fc({image, image_pre})
  local SR_fc_o = SR_fc(h)
  local hr_patch = SRnet({patch, patch_pre, SR_patch_fc_o, SR_img_fc_o, SR_fc_o})
  if opt.residual then hr_patch = nn.Tanh()(nn.CAddTable()({hr_patch,patch_pre})) end
  local image_next = nn.SpatialGlimpse_inverse(opt.glimpsePatchSize, nil)({image_pre,hr_patch,loc})
  
  nngraph.annotateNodes()
  model = nn.gModule({loc_prev,image_pre,visited_map_pre,onesTensor,image}, {loc, image_next, visited_map})
  model:apply(weights_init)
  model.name = 'fullmodel'
  model = nn.Recursor(model, opt.rho)
 
  print('SR_patch_fc',SR_patch_fc)
  print('SR_img_fc',SR_img_fc)
  print('SR_fc',SR_fc)
 end
 
---------------------------------------------------------------------------

optimState = {
learningRate = opt.lr,
beta1 = opt.beta1,
}

model:forget()
local parameters, gradParameters = model:getParameters()
thin_model = model:sharedClone() -- used for save checkpoint
local a, b = thin_model:getParameters()
print(parameters:nElement())
print(gradParameters:nElement())



 -- train
epoch = 0
while epoch < 1 do
  epoch = epoch+1
  gradParameters:zero()
  model:forget()   
  --fetch data
  highRes, idLabel = data:getBatch()
  print("highRes, idLabel",highRes, idLabel)
  lowRes = highRes:clone()
  for imI = 1, highRes:size(1) do
    temp = image.scale(highRes[imI], lowResSize[2], lowResSize[1])
    lowRes[imI] = image.scale(temp, highResSize[2], highResSize[1], 'bicubic')
  end
  highRes = highRes
  lowRes = lowRes
  idLabel = idLabel

  local zero_loc = torch.zeros(opt.batchSize,2)
  local zero_dummy = torch.zeros(opt.batchSize,1)
  local ones = torch.ones(opt.batchSize,1,opt.glimpsePatchSize[1],opt.glimpsePatchSize[2])
  local visited_map0 = torch.zeros(opt.batchSize,1,highResSize[1],highResSize[2])
  zero_loc = zero_loc
  zero_dummy = zero_dummy
  ones = ones
  visited_map0 = visited_map0

  --print('visited_map0',visited_map0)

  local dl = {}
  local inputs = {}
  outputs = {}
  NewValue={}
  gt = {}
  err_l = 0
  err_g = 0
  
  -- input: {loc_prev, image_pre, visited_map_prev, ones, image}
  -- output: {loc, image_next, visited_map_next}


	inputs = {zero_loc, lowRes, visited_map0, ones, lowRes}

	outputs = model:forward(inputs)
	print('outputs',outputs[2][1])
	pos =  (((outputs[1] +1 ) * (128 - 0)) / (1 +1)) + 0
	print('pos',pos)
	NewValue = (((outputs[2][1] +1) * (1 - 0)) / (1 +1)) + 0

	print(NewValue)
	image.save('hr_patch'..tostring(epoch)..'.png',NewValue)
 	--image.display(NewValue)

end
