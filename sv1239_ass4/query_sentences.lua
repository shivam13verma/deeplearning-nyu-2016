--Deep Learning: HW 4
--query_sentences.lua
--Author: Shivam Verma

require 'nngraph'
require 'xlua'
require 'io'

function parse_cmdline()
	local opt = lapp[[
		--model_dir	(default "./")
		--model_name	(default "model.hw4.net")
		--layers	(default 2)
		--batch_size	(default 20)
		--data_dir	(default "./data/")
	]]
	return opt
end

params = parse_cmdline()

--some base.lua stuff here
function g_disable_dropout(node)
    if type(node) == "table" and node.__typename == nil then
        for i = 1, #node do
            node[i]:apply(g_disable_dropout)
        end
        return
    end
    if string.match(node.__typename, "Dropout") then
        node.train = false
    end
end

function g_enable_dropout(node)
    if type(node) == "table" and node.__typename == nil then
        for i = 1, #node do
            node[i]:apply(g_enable_dropout)
        end
        return
    end
    if string.match(node.__typename, "Dropout") then
        node.train = true
    end
end

function g_cloneManyTimes(net, T)
    local clones = {}
    local params, gradParams = net:parameters()
    local mem = torch.MemoryFile("w"):binary()
    mem:writeObject(net)
    for t = 1, T do
        -- We need to use a new reader for each clone.
        -- We don't want to use the pointers to already read objects.
        local reader = torch.MemoryFile(mem:storage(), "r"):binary()
        local clone = reader:readObject()
        reader:close()
        local cloneParams, cloneGradParams = clone:parameters()
        for i = 1, #params do
            cloneParams[i]:set(params[i])
            cloneGradParams[i]:set(gradParams[i])
        end
        clones[t] = clone
        collectgarbage()
    end
    mem:close()
    return clones
end

function g_init_gpu(args)
    local gpuidx = args
    gpuidx = gpuidx[1] or 1
    print(string.format("Using %s-st gpu", gpuidx))
    cutorch.setDevice(gpuidx)
    g_make_deterministic(1)
end

function g_make_deterministic(seed)
    torch.manualSeed(seed)
    cutorch.manualSeed(seed)
    torch.zeros(1, 1):cuda():uniform()
end

function g_replace_table(to, from)
    assert(#to == #from)
    for i = 1, #to do
        to[i]:copy(from[i])
    end
end

function g_f3(f)
    return string.format("%.3f", f)
end

function g_d(f)
    return string.format("%d", torch.round(f))
end
local stringx = require('pl.stringx')
local file = require('pl.file')

local ptb_path = params.data_dir

local trainfn = ptb_path .. "ptb.train.txt"
local testfn  = ptb_path .. "ptb.test.txt"
local validfn = ptb_path .. "ptb.valid.txt"

local vocab_idx = 0
local vocab_map = {}

-- Stacks replicated, shifted versions of x_inp
-- into a single matrix of size x_inp:size(1) x batch_size.

local function replicate(x_inp, batch_size)
    local s = x_inp:size(1)
    local x = torch.zeros(torch.floor(s / batch_size), batch_size)
    for i = 1, batch_size do
        local start = torch.round((i - 1) * s / batch_size) + 1
        local finish = start + x:size(1) - 1
        x:sub(1, x:size(1), i, i):copy(x_inp:sub(start, finish))
    end
    return x
end

local function load_data(fname)
    local data = file.read(fname)
    data = stringx.replace(data, '\n', '<eos>')
    data = stringx.split(data)
    --print(string.format("Loading %s, size of data = %d", fname, #data))
    local x = torch.zeros(#data)
    for i = 1, #data do
        if vocab_map[data[i]] == nil then
            vocab_idx = vocab_idx + 1
            vocab_map[data[i]] = vocab_idx
        end
        x[i] = vocab_map[data[i]]
    end
    return x
end

local function traindataset(batch_size, char)
   local x = load_data(trainfn)
   x = replicate(x, batch_size)
   return x
end

-- Intentionally we repeat dimensions without offseting.
-- Pass over this batch corresponds to the fully sequential processing.
local function testdataset(batch_size)
    if testfn then
        local x = load_data(testfn)
        x = x:resize(x:size(1), 1):expand(x:size(1), batch_size)
        return x
    end
end

local function validdataset(batch_size)
    local x = load_data(validfn)
    x = replicate(x, batch_size)
    return x
end


--data.lua stuff ends here
--
--
function transfer_data(x)
	if gpu then
		return x:cuda()
	else
		return x
	end
end


function reset_state(state)
    state.pos = 1
    if model ~= nil and model.start_s ~= nil then
        for d = 1, 2*layers do
            model.start_s[d]:zero()
        end
    end
end

--runs a query on the model
function run_sentence_query(length, indices)
  --initialize 
  for d = 1, 2 * params.layers do
      model.start_s[d]:zero()
   end

   --disable dropout in test_mode
   g_disable_dropout(model.rnns)

   -- Fill indices with 1's if not present.
   local sentence_ind = torch.ones(indices:size(1)+length)
   for i = 1, indices:size(1) do
      sentence_ind[i] = indices[i]
   end

   -- Resize input according to batch..
   local sentence_inputs = sentence_ind:resize(sentence_ind:size(1), 1):expand(sentence_ind:size(1), params.batch_size)

   g_replace_table(model.s[0], model.start_s)
   for i = 1, sentence_inputs:size(1)-1 do
      local x = sentence_inputs[i]
      local y = sentence_inputs[i+1]
	--returns log probability score
      _, model.s[1], log_proby = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
      if i >= indices:size(1) then
         -- Sample from the predictions as suggested.
         local pred_ind = torch.multinomial(torch.exp(log_proby), 1)
         sentence_inputs[i+1] = pred_ind
      end
      g_replace_table(model.s[0], model.s[1])
   end
   g_enable_dropout(model.rnns)
   --return sentence indices correlating to dict
   return sentence_ind
end


--reads query from stdin
function read_query()
  local line = io.read("*line")
  if line == nil then error({code="EOF"}) end
  line = string.lower(line)
  line = stringx.split(line)
  local sentence_length = tonumber(line[1])
  if sentence_length == nil then
     error({code="init"})
  end
  word_indices = {}
  for i = 2,#line do
     if vocab_map[line[i]] == nil then
        error({code="vocab", word = line[i]})
     end
  end
  return line
end

--Inverts vocab_map to obtain predicted words from the RNN.
function vocab_map_inverter()
	local vocab_ind = {}
	for k,v in pairs(vocab_map) do
		vocab_ind[v]=k --invert key-val pair
	end
	return vocab_ind
end

--returns completed sentence. Calls another function which does the heavy work.
function run_sentence_completer()
   local vocab_idx_map = vocab_map_inverter()

    while true do
      print("Query should have form: <len> <word1> <word2> ...")
      local ok, line = pcall(read_query)
      if not ok then
	 --check if end of file reached
         if line.code == "EOF" then
            break 
	 --check if word in vocab_dict
         elseif line.code == "vocab" then
            print("Sorry, this word is not present in vocab_dict: "..line.word)
         elseif line.code == "init" then
            print("Please type an integer first!")
         else
            print(line)
            print("Sorry, something went wrong. Try again.")
         end
      else
         local length = tonumber(line[1])
         local indices = torch.zeros(#line - 1)
         for i = 2,#line do
            indices[i-1] = vocab_map[line[i]]
         end
         pred_indices = run_sentence_query(length, indices)
         words = ""
	 --append words with highest proby
         for i = 1,pred_indices:size(1) do
            words = words .. vocab_idx_map[pred_indices[i][1]] .. " "
         end
         print(words)
      end
      print("NOTE: To exit, press ctrl-C twice")
   end
end


state_train = {data=traindataset(params.batch_size)}
state_valid =  {data=validdataset(params.batch_size)}
state_test =  {data=testdataset(params.batch_size)}

print("==> Loading model")
model = torch.load(params.model_dir .. params.model_name)
print("Model " .. params.model_name .. " has been loaded from " .. params.model_dir)
run_sentence_completer()
