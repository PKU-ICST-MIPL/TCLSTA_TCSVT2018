
require 'nn'
require 'torch'
require 'cunn'
require 'optim'
require 'misc.word_level'
require 'misc.recursive_atten'
require 'misc.optim_updates'
local utils = require 'misc.utils'

-------------------------------------------------------------------------------
-- Input arguments and options
-------------------------------------------------------------------------------

cmd = torch.CmdLine()
cmd:text()
cmd:text('Train a Visual Question Answering model')
cmd:text()
cmd:text('Options')

-- Data input settings

cmd:option('-start_from', '', 'path to a model checkpoint to initialize model weights from. Empty = don\'t')
cmd:option('-co_atten_type', 'Alternating', 'co_attention type. Parallel or Alternating, alternating trains more faster than parallel.')
cmd:option('-feature_type', 'Residual', 'VGG or Residual')


cmd:option('-hidden_size',2048,'the hidden layer size of the model.')
cmd:option('-rnn_size',512,'size of the rnn in number of hidden nodes in each layer')
cmd:option('-batch_size',20,'what is theutils batch size in number of images per batch? (there will be x seq_per_img sentences)')
cmd:option('-output_size', 1000, 'number of output answers')
cmd:option('-rnn_layers',2,'number of the rnn layer')


-- Optimization
cmd:option('-optim','rmsprop','what update to use? rmsprop|sgd|sgdmom|adagrad|adam')
cmd:option('-learning_rate',5e-4,'learning rate')
cmd:option('-learning_rate_decay_start', 0, 'at what iteration to start decaying learning rate? (-1 = dont)')
cmd:option('-learning_rate_decay_every', 100, 'every how many epoch thereafter to drop LR by 0.1?')
cmd:option('-optim_alpha',0.99,'alpha for adagrad/rmsprop/momentum/adam')
cmd:option('-optim_beta',0.995,'beta used for adam')
cmd:option('-optim_epsilon',1e-8,'epsilon that goes into denominator in rmsprop')
cmd:option('-max_iters', -1, 'max number of iterations to run for (-1 = run forever)')
cmd:option('-iterPerEpoch', 1200)

-- Evaluation/Checkpointing
cmd:option('-save_checkpoint_every', 6000, 'how often to save a model checkpoint?')
cmd:option('-checkpoint_path', 'save/', 'folder to save checkpoints into (empty = this folder)')

-- Visualization
cmd:option('-losses_log_every', 10, 'How often do we save losses, for inclusion in the progress dump? (0 = disable)')

-- misc
cmd:option('-id', '0', 'an id identifying this run/job. used in cross-val and appended when writing progress files')
cmd:option('-backend', 'cudnn', 'cudnn')
cmd:option('-gpuid', 0, 'which gpu to use. -1 = use CPU')
cmd:option('-seed', 123, 'random number generator seed to use')

cmd:text()

-------------------------------------------------------------------------------
-- Basic Torch initializations
-------------------------------------------------------------------------------
local opt = cmd:parse(arg)
torch.manualSeed(opt.seed)
print(opt)
torch.setdefaulttensortype('torch.FloatTensor') -- for CPU

if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then 
  require 'cudnn' 
  end
  cutorch.setDevice(opt.gpuid+1) -- note +1 because lua is 1-indexed
  cutorch.manualSeed(opt.seed)
end

opt = cmd:parse(arg)

-------------------------------------------------------------------------------
-- Read feature data
-------------------------------------------------------------------------------

local spaTrainFeatureLabels = torch.load('data/ucf101_train01_frame_sp25_res50_cvgj_spatial_temporal_global_pool_nxdx25.t7') 
local spaTrainData = spaTrainFeatureLabels.featMats
local spaTrainTarget = spaTrainFeatureLabels.labels
local spaTrain = torch.Tensor(spaTrainData:size(1),25,2048)
for pi = 1, spaTrainData:size(1) do   
  for y = 1,  spaTrainData:size(3) do
    spaTrain[{pi,y,{}}] = spaTrainData[{{pi},{},{y}}]
  end
end
print(spaTrain:size())

local tempTrainFeatureLabels = torch.load('data/ucf101_train01_optflow_sp25_res50_cvgj_spatial_temporal_global_pool_nxdx25.t7')
local tempTrainData = tempTrainFeatureLabels.featMats
local tempTrainTarget = tempTrainFeatureLabels.labels
local tempTrain = torch.Tensor(tempTrainData:size(1),25,2048)
for pi = 1, tempTrainData:size(1) do   
  for y = 1,  tempTrainData:size(3) do
    tempTrain[{pi,y,{}}] = tempTrainData[{{pi},{},{y}}]
  end
end
print(tempTrain:size())

local spaTestFeatureLabels = torch.load('data/ucf101_test01_frame_sp25_res50_cvgj_spatial_temporal_global_pool_nxdx25.t7') 
local spaTestData = spaTestFeatureLabels.featMats
local spaTestTarget = spaTestFeatureLabels.labels
local spaTest = torch.Tensor(spaTestData:size(1),25,2048)
for pi = 1, spaTestData:size(1) do   
  for y = 1,  spaTestData:size(3) do
    spaTest[{pi,y,{}}] = spaTestData[{{pi},{},{y}}]
  end
end
print(spaTest:size())

local tempTestFeatureLabels = torch.load('data/ucf101_test01_frame_sp25_res50_cvgj_spatial_temporal_global_pool_nxdx25.t7')
local tempTestData = tempTestFeatureLabels.featMats
local tempTestTarget = tempTestFeatureLabels.labels
local tempTest = torch.Tensor(tempTestData:size(1),25,2048)
for pi = 1, tempTestData:size(1) do   
  for y = 1,  tempTestData:size(3) do
    tempTest[{pi,y,{}}] = tempTestData[{{pi},{},{y}}]
  end
end
print(tempTest:size())

print("load data succeed")

------------------------------------------------------------------------
--Design Parameters and Network Definitions
------------------------------------------------------------------------
local protos = {}
print('Building the model...')
-- intialize language model
local loaded_checkpoint
local lmOpt
if string.len(opt.start_from) > 0 then
  local start_path = path.join(opt.checkpoint_path .. '_' .. opt.co_atten_type ,  opt.start_from)
  loaded_checkpoint = torch.load(start_path)
  lmOpt = loaded_checkpoint.lmOpt
else
  lmOpt = {}
  lmOpt.hidden_size = opt.hidden_size
  lmOpt.rnn_size = opt.rnn_size
  lmOpt.num_layers = opt.rnn_layers
  lmOpt.dropout = 0.5
  lmOpt.batch_size = opt.batch_size
  lmOpt.output_size = opt.rnn_size
  lmOpt.atten_type = opt.co_atten_type
  lmOpt.feature_type = opt.feature_type
  lmOpt.seq_length = 25
end
protos.word = nn.word_level(lmOpt)
protos.atten = nn.recursive_atten()
protos.crit = nn.CrossEntropyCriterion()
-- ship everything to GPU, maybe

if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end

local wparams, grad_wparams = protos.word:getParameters()
local aparams, grad_aparams = protos.atten:getParameters()


if string.len(opt.start_from) > 0 then
  print('Load the weight...')
  wparams:copy(loaded_checkpoint.wparams)
  aparams:copy(loaded_checkpoint.aparams)
end

print('total number of parameters in word_level: ', wparams:nElement())
assert(wparams:nElement() == grad_wparams:nElement())


print('total number of parameters in recursive_attention: ', aparams:nElement())
assert(aparams:nElement() == grad_aparams:nElement())

collectgarbage() 

-------------------------------------------------------------------------------
-- Validation evaluation
-------------------------------------------------------------------------------
local function eval_split(batch_size)

  protos.word:evaluate()
  protos.atten:evaluate()


  local n = 0
  local loss_sum = 0
  local loss_evals = 0
  local right_sum = 0
  local predictions = {}
  local total_num = spaTestTarget:size(1)
  local start = 1

  while true do
    local answer
    local streams
    local images
    -- ship the data to cuda

    if (total_num - start) < (batch_size-1) then
      batch_size = total_num - start + 1
    end
    if opt.gpuid >= 0 then
      answer = spaTestTarget:narrow(1, start, batch_size):cuda()
      streams = tempTest:narrow(1, start, batch_size):cuda()
      images = spaTest:narrow(1, start, batch_size):cuda()

    end
    n = n + answer:size(1)
    start = start + answer:size(1)
    
    local word_feat, img_feat, w_ques, w_img, mask = unpack(protos.word:forward({streams, images}))

    local feature_ensemble = {w_ques, w_img}
    local out_feat = protos.atten:forward(feature_ensemble) --final classification

  -- forward the language model criterion
    local loss = protos.crit:forward(out_feat, answer)

    local tmp,pred=torch.max(out_feat,2)

    for i = 1, pred:size()[1] do

      if pred[i][1] == answer[i] then
        right_sum = right_sum + 1
      end
    end

    loss_sum = loss_sum + loss
    loss_evals = loss_evals + 1
    if n >= total_num then break end
  end

  return loss_sum/loss_evals, right_sum / total_num
end
-------------------------------------------------------------------------------
-- Loss function
-------------------------------------------------------------------------------
local iter = 0
local function lossFun(batch_size)
  protos.word:training()
  grad_wparams:zero()  

  protos.atten:training()
  grad_aparams:zero()

  ----------------------------------------------------------------------------
  -- Forward pass
  -----------------------------------------------------------------------------
  -- get batch of data  
  local answer
  local streams
  local images
  trainIndex = torch.LongTensor(batch_size):zero()
  
  for i=1, batch_size do  

   trainIndex[i] =  math.random(spaTrainTarget:size(1))  
  end  
  if opt.gpuid >= 0 then
    answer = spaTrainTarget:index(1,trainIndex):cuda()
    streams = tempTrain:index(1,trainIndex):cuda()
    images = spaTrain:index(1,trainIndex):cuda()
  end

  local word_feat, img_feat, w_ques, w_img, mask = unpack(protos.word:forward({streams, images}))
  --print(word_feat:size())
  --print(img_feat:size())
  --print(w_ques:size())
  --print(w_img:size())
  local feature_ensemble = {w_ques, w_img}
  local out_feat = protos.atten:forward(feature_ensemble) --final classfication

  -- forward the language model criterion
  local loss = protos.crit:forward(out_feat, answer)
  -----------------------------------------------------------------------------
  -- Backward pass
  -----------------------------------------------------------------------------
  -- backprop criterion
  local dlogprobs = protos.crit:backward(out_feat, answer)
  
  local d_w_ques, d_w_img = unpack(protos.atten:backward(feature_ensemble, dlogprobs))
  
  local dummy = protos.word:backward({streams, images}, {d_w_ques, d_w_img})

  -----------------------------------------------------------------------------
  -- and lets get out!
  local stats = {}
  stats.dt = dt
  local losses = {}
  losses.total_loss = loss
  return losses, stats

end


local w_optim_state = {}
local a_optim_state = {}
local loss0
local loss_history = {}
local accuracy_history = {}
local learning_rate_history = {}
local ave_loss = 0
local learning_rate = opt.learning_rate
local timer = torch.Timer()
local decay_factor = math.exp(math.log(0.1)/opt.learning_rate_decay_every/opt.iterPerEpoch)
math.randomseed(os.time())  
while true do
  -- eval loss/gradient
  local losses, stats = lossFun(256)
  ave_loss = ave_loss + losses.total_loss
  -- decay the learning rate
  if iter > opt.learning_rate_decay_start and opt.learning_rate_decay_start >= 0 then
    learning_rate = learning_rate * decay_factor -- set the decayed rate
  end

  if iter % opt.losses_log_every == 0 then
    ave_loss = ave_loss / opt.losses_log_every
    loss_history[iter] = losses.total_loss 
    accuracy_history[iter] = ave_loss
    learning_rate_history[iter] = learning_rate

    print(string.format('iter %d: %f, %f, %f, %f', iter, losses.total_loss, ave_loss, learning_rate, timer:time().real))

    ave_loss = 0
    local val_loss, val_accu = eval_split(10)
    print('test loss: ', val_loss, 'accuracy ', val_accu, 'time ',timer:time().real)
  end
  --[[if iter % 500 == 0 then
    local val_loss, val_accu = eval_split(10)
    print('test loss: ', val_loss, 'accuracy ', val_accu)
  end]]
  if opt.optim == 'rmsprop' then
    rmsprop(wparams, grad_wparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, w_optim_state)
    rmsprop(aparams, grad_aparams, learning_rate, opt.optim_alpha, opt.optim_epsilon, a_optim_state)
  else
    error('bad option opt.optim')
  end
  --print(wparams:size()) 
  --print(aparams:size()) 
  --print(grad_wparams:size()) 
  --print(grad_aparams:size()) 
  iter = iter + 1

end
