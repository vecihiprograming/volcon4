--[[
    Copyright (c) 2015-present, Facebook, Inc.
    All rights reserved.

    This source code is licensed under the BSD-style license found in the
    LICENSE file in the root directory of this source tree. An additional grant
    of patent rights can be found in the PATENTS file in the same directory.
]]--

-- Heavily moidifed by Carl to make it simpler

require 'torch'
require 'image'
tds = require 'tds'
torch.setdefaulttensortype('torch.FloatTensor')
local class = require('pl.class')

local dataset = torch.class('dataLoader')
local sampleSize = opt.loadSize --{128x96}

-- this function reads in the data files
function dataset:__init(args)
  for k,v in pairs(args) do self[k] = v end

  assert(self.frameSize > 0)

  -- read text file consisting of frame directories and counts of frames
  self.data = tds.Vec()
  print('reading ' .. args.data_list)
  for line in io.lines(args.data_list) do 
	self.data:insert(line)
  end
  --print(self.data)
  print('found ' .. #self.data .. ' videos')
end

function dataset:size()
  return #self.data
end

-- converts a table of samples (and corresponding labels) to a clean tensor
function dataset:EAtableToOutput(dataTable, extraTable)
   --print('--function dataset:tableToOutput(dataTable, extraTable)')
   local data, scalarLabels, labels
   local quantity = #dataTable
   assert(dataTable[1]:dim() == 4)

   data = torch.Tensor(quantity,self.frameSize, 3, self.oW, self.oH)
   --print('sil')
   for i=1,#dataTable do
      data[i]:copy(dataTable[i])
   end
   return data, extraTable
end

-- converts a table of samples (and corresponding labels) to a clean tensor
function dataset:tableToOutput(dataTable, extraTable)
   --print('--function dataset:tableToOutput(dataTable, extraTable)')
   local data, scalarLabels, labels
   local quantity = #dataTable
   assert(dataTable[1]:dim() == 4)

   data = torch.Tensor(quantity, 3, self.frameSize, self.oW, self.oH)
   --print('sil')
   for i=1,#dataTable do
      data[i]:copy(dataTable[i])
   end
   return data, extraTable
end

-- sampler, samples with replacement from the training set.
function dataset:sample(quantity)
   print('donkey_video2 function dataset:sample(quantity)')
   assert(quantity)
   local dataTable = {}
   local extraTable = {}
   for i=1,quantity do
      local idx = torch.random(1, #self.data)
      local data_path = self.data[idx]--klasör
	  --print(data_path)

      local out = self:BBtrainHook(data_path)
	  --print(out)
      table.insert(dataTable, out)
      table.insert(extraTable, self.data[idx])
   end
   
   --print(self:tableToOutput(dataTable,extraTable))
   return self:tableToOutput(dataTable,extraTable)
end

-- sampler, samples with replacement from the training set.
function dataset:EAsample(quantity)
   print('donkey_video2 function dataset:EAsample(quantity)')
   assert(quantity)
   local dataTable = {}
   local extraTable = {}
   -- local dataTable2 = {}
   -- local extraTable2 = {}   
   for i=1,quantity do
      local idx = torch.random(1, #self.data)
      local data_path = self.data[idx]--klasör
	  --print(data_path)

      local out = self:EAtrainHook(data_path)
	-- local out2 = self:BBtrainHook(data_path)
	  --print(out)
      table.insert(dataTable, out)
      -- table.insert(dataTable2, out2)
      table.insert(extraTable, self.data[idx])
      -- table.insert(extraTable2, self.data[idx])
   end
   
   --print(self:EAtableToOutput(dataTable))
   return self:EAtableToOutput(dataTable,extraTable)
end

-- gets data in a certain range
function dataset:get(start_idx,stop_idx)
   local dataTable = {}
   local extraTable = {}
   for idx=start_idx,stop_idx do
      local data_path = self.data_root .. '/' .. self.data[idx]

      local out = self:trainHook(data_path)
      table.insert(dataTable, out)
      table.insert(extraTable, self.data[idx])
   end
   return self:tableToOutput(dataTable,extraTable)

end


function dataset:BBtrainHook(path)
-- bunu volumetric icin hazirliyorum
  print('function dataset:BBtrainHook(path)')
  collectgarbage()
  
  local out = torch.zeros(3, self.frameSize, self.oW, self.oH)
	
  --print(path)

  for fr=1,self.frameSize do
	local ok,input = pcall(image.load, path .. '/' .. 'im' .. fr .. '.png', 3, 'float') 

	--print(input)
	-- hook ları burayı ekle ınput
	out[{ {}, fr, {}, {} }]:copy(input)
  end
  --print(out)
  
  out:mul(2):add(-1) -- make it [0, 1] -> [-1, 1]
  
  return out
end


function dataset:EAtrainHook(path)
-- bunu image dizisi icin hazirliyorum
  print('function dataset:EAtrainHook(path)')
  print(path)
  collectgarbage()
  
  local out = torch.zeros(self.frameSize, 3, self.oW, self.oH)
	
  --print(path)

  for fr=1,self.frameSize do
	local ok,input = pcall(image.load, path .. '/' .. 'im' .. fr .. '.png', 3, 'float') 
	--print(input,'sdf')
	-- hook ları burayı ekle ınput
	local iW = input:size(2)
	local iH = input:size(3)
	--print('iW iH',iW,iH)
	local oW = self.oW --128
	local oH = self.oH --96
	--print('oW oH',oW,oH)
	if iH > oH and iW > oW then
	  --print('ifin içi')
	  local h1 = math.ceil(torch.uniform(1e-2, iH-oH))
	  local w1 = math.ceil(torch.uniform(1e-2, iW-oW))
	  out = image.crop(input, w1, h1, w1 + oW, h1 + oH)
	else
		--print('ifin dışı')
		out[{ fr, {}, {}, {} }]:copy(input)
	end
	
	assert(out:size(3) == oW)
	assert(out:size(4) == oH)
  end
	out:mul(2):add(-1) -- make it [0, 1] -> [-1, 1]
  return out
end


-- function to load the image, jitter it appropriately (random crops etc.)
function dataset:trainHook(path)
  print('donkey_video2 function dataset:trainHook(path)')
  collectgarbage()

  local oW = self.fineSize
  local oH = self.fineSize 
  local h1
  local w1

  local out = torch.zeros(3, self.frameSize, oW, oH)

  local ok,input = pcall(image.load, path, 3, 'float') 
  if not ok then
     print('warning: failed loading: ' .. path)
     return out
  end

  local count = input:size(2) / opt.loadSize
  local t1 = 1
  
  for fr=1,self.frameSize do
    local off 
    if fr <= count then 
      off = (fr+t1-2) * opt.loadSize+1
    else
      off = (count+t1-2)*opt.loadSize+1 -- repeat the last frame
    end
    local crop = input[{ {}, {off, off+opt.loadSize-1}, {} }]
    out[{ {}, fr, {}, {} }]:copy(image.scale(crop, opt.fineSize, opt.fineSize))
  end

  out:mul(2):add(-1) -- make it [0, 1] -> [-1, 1]

  -- subtract mean
  for c=1,3 do
    out[{ c, {}, {} }]:add(-self.mean[c])
  end

  return out
end

-- data.lua expects a variable called trainLoader
trainLoader = dataLoader(opt)
