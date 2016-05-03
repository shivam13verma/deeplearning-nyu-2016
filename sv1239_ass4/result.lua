---Deep Learning: HW 4
--result.lua
--Author: Shivam Verma

require('nngraph')
require ('xlua')
--ptb = require('data')

function parse_cmdline()
	local opt = lapp[[
		--model_dir	(default "./")			model is in this directory
		--model_name	(default "model.hw4.net")	name of the model
		--data_dir	(default "./data/")			data is in this directory
	]]
	return opt
end

params = parse_cmdline()


local stringx = require('pl.stringx')
local file = require('pl.file')

--data folder must be in current directory
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
--
--return {traindataset=traindataset,
--        testdataset=testdataset,
--        validdataset=validdataset,
--        vocab_map=vocab_map}

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

--real work starts here
local batch_size = 20
local layers = 2

function reset_state(state)
    state.pos = 1
    if model ~= nil and model.start_s ~= nil then
        for d = 1, 2*layers do
            model.start_s[d]:zero()
        end
    end
end

function run_test()
    reset_state(state_test)
    g_disable_dropout(model.rnns)
    local perp = 0
    local len = state_test.data:size(1)
    
    -- no batching here
    g_replace_table(model.s[0], model.start_s)
    for i = 1, 100 do -- (len - 1) do
        
	xlua.progress(i, 100)
	--xlua.progress(i, len-1)
	local x = state_test.data[i]
        local y = state_test.data[i + 1]
        perp_tmp, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
	perp = perp + perp_tmp[1]
        g_replace_table(model.s[0], model.s[1])
    end
    print("Test set perplexity : " .. g_f3(torch.exp(perp / 100)))--(len - 1))))
    g_enable_dropout(model.rnns)
end

--model_dir="/scratch/sv1239/projects/deeplearning/hw4/gru/"
state_train = {data=traindataset(batch_size)}
state_valid =  {data=validdataset(batch_size)}
state_test =  {data=testdataset(batch_size)}

print("==> Loading model")
model = torch.load(params.model_dir .. params.model_name)
run_test()
--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the Apache 2 license found in the
--  LICENSE file in the root directory of this source tree. 
--

