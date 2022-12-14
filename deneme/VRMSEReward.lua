------------------------------------------------------------------------
--[[ VRMSEReward ]]--
-- Variance reduced classification reinforcement criterion.
-- input : {class prediction, baseline reward}
-- Reward is 1 for success, Reward is 0 otherwise.
-- reward = scale*(Reward - baseline) where baseline is 2nd input element
-- Note : for RNNs with R = 1 for last step in sequence, encapsulate it
-- in nn.ModuleCriterion(VRMSEReward, nn.SelectTable(-1))
------------------------------------------------------------------------
local VRMSEReward, parent = torch.class("nn.VRMSEReward", "nn.Criterion")

function VRMSEReward:__init(module, scale, areaScale, criterion)
   parent.__init(self)
   self.module = module -- so it can call module:reinforce(reward)
   self.scale = scale or 1 -- scale of reward
   self.areaScale = areaScale or 4 -- scale of reward
   self.errorC = nn.MSECriterion()
   -- self.fillOp = nn.SpatialGlimpse_inverse(opt.glimpsePatchSize)
   -- self.fill = torch.ones(opt.batchSize,opt.nc,opt.glimpsePatchSize[1],opt.glimpsePatchSize[2])
   self.criterion = criterion or nn.MSECriterion() -- baseline criterion
   self.sizeAverage = true
   self.gradInput = {torch.Tensor()}
end

function VRMSEReward:updateOutput(inputTable, target) --forward
print('VRMSEReward:updateOutput içindesin')
print('1. Giriş = inputTable',inputTable)
print('2. Giriş = target',#target) -- target=highRes
   assert(torch.type(inputTable) == 'table')
   local map = inputTable[2] -- batchsize*1*128*128 (adamınkinde) visited_map_next
   local input = inputTable[1]-- batchsize*3*128*128 image_next
   
   print('map=inputTable[2]==>',#map) -- visited_map_next
   print('input=inputTable[1]==>',#input)--image_next

   -- reward = mse
   self.reward = input:clone()
   self.reward:add(-1, target):pow(2)
   assert(self.reward:dim() == 4)  -- burada 4 oluşu batch*3*128*128 yani 4 boy olusu
   for i = 4,2,-1 do
      self.reward = self.reward:sum(i)
   end
   self.reward:resize(self.reward:size(1))
   self.reward:div(-input:size(3)*input:size(4))
   self.reward:add(4) -- pixel ~ [-1, 1], thus error^2 are <= 4
   local area = map:sum(4):sum(3):sum(2):div(opt.highResSize[1]*opt.highResSize[2])-- tüm pikselleri topladı 128*128 e böldü tensör elde etti
   area = area:view(-1) -- 1x1x1x2 boyutunu direkt 2 yaptı
   print('area',area)
   self.reward:add(self.areaScale,area) -- 4 = area reward scale

   self.output = self.errorC:forward(input,target)--errorC=MSECriterion  //  image_next ile highRes MSE yapıo
   
   return self.output
end

function VRMSEReward:updateGradInput(inputTable, target) --backward
print('VRMSEReward:updateGradInput içindesin')
   local input = inputTable[1] -- loc
   local baseline =inputTable[3] -- viseted map
   
   -- reduce variance of reward using baseline
   self.vrReward = self.vrReward or self.reward.new()
   self.vrReward:resizeAs(self.reward):copy(self.reward)
   self.vrReward:add(-1, baseline)
   if self.sizeAverage then
      self.vrReward:div(input:size(1))
   end
   -- broadcast reward to modules
   self.module:reinforce(self.vrReward)  
   
   -- zero gradInput (this criterion has no gradInput for class pred)
   self.gradInput[1]:resizeAs(input):zero() --gradInput = torch.Tensor()
   self.gradInput[1] = self.gradInput[1]
   -- self.gradInput[1] = self.errorC:backward(input,target)

   -- learn the baseline reward
   self.gradInput[3] = self.criterion:backward(baseline, self.reward)
   self.gradInput[3] = self.gradInput[3]
   return self.gradInput
end

function VRMSEReward:type(type)
   local module = self.module
   self.module = nil
   local ret = parent.type(self, type)
   self.module = module
   return ret
end