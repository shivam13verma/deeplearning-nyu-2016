--Deep Learning HW 4
--main.lua
--Author: Shivam Verma

gpu = false
if gpu then
    require 'cunn'
    print("Running on GPU") 
    
else
    require 'nn'
    print("Running on CPU")
end

stringx = require('pl.stringx')
require('nngraph')
require('base') 
ptb = require('data')


function parse_cmdline()
   local opt = lapp[[
      --batch_size            (default 20)         
      --seq_length            (default 20)        
      --layers                (default 2)        
      --decay                 (default 2)       
      --rnn_size              (default 200)    
      --dropout               (default 0)
      --init_weight           (default 0.1)   
      --lr                    (default 1)    
      --vocab_size            (default 10000)  
      --max_epoch             (default 4)     
      --max_max_epoch         (default 13)   
      --max_grad_norm         (default 5)   
      --model_type            (default lstm)       .
      --model_dir             (default "./")
      --model_name            (default "model.hw4.net")
   ]]
   return opt

end

params = parse_cmdline()

function transfer_data(x)
    if gpu then
        return x:cuda()
    else
        return x
    end
end

model = {}

local function lstm(x, prev_c, prev_h)
    -- Calculate all four gates in one go
    local i2h              = nn.Linear(params.rnn_size, 4*params.rnn_size)(x)
    local h2h              = nn.Linear(params.rnn_size, 4*params.rnn_size)(prev_h)
    local gates            = nn.CAddTable()({i2h, h2h})

    -- Reshape to (batch_size, n_gates, hid_size)
    -- Then slize the n_gates dimension, i.e dimension 2
    local reshaped_gates   =  nn.Reshape(4,params.rnn_size)(gates)
    local sliced_gates     = nn.SplitTable(2)(reshaped_gates)

    -- Use select gate to fetch each gate and apply nonlinearity
    local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
    local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
    local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
    local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

    local next_c           = nn.CAddTable()({
        nn.CMulTable()({forget_gate, prev_c}),
        nn.CMulTable()({in_gate,     in_transform})
    })
    local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

    return next_c, next_h
end

--second param is dummy. Needed since create_network can build both lstm's and gru's.
function gru(x,_,prev_h)

  local i2h = nn.Linear(params.rnn_size, 3 * params.rnn_size)(x)
  local h2h = nn.Linear(params.rnn_size, 3 * params.rnn_size)(prev_h)
  local gates = nn.CAddTable()({
        nn.Narrow(2, 1, 2 * params.rnn_size)(i2h),
        nn.Narrow(2, 1, 2 * params.rnn_size)(h2h),})
  gates = nn.SplitTable(2)(nn.Reshape(2, params.rnn_size)(gates))
  local reset_gate = nn.Sigmoid()(nn.SelectTable(1)(gates))
  local update_gate = nn.Sigmoid()(nn.SelectTable(2)(gates))
  local output = nn.Tanh()(nn.CAddTable()({
                                 nn.Narrow(2, 2 * params.rnn_size+1, params.rnn_size)(i2h),
                                 nn.CMulTable()({
                                       reset_gate,
                                       nn.Narrow(2, 2 * params.rnn_size+1, params.rnn_size)(h2h),})}))
  local next_h = nn.CAddTable()({ prev_h,
                                 nn.CMulTable()({ update_gate,
                                                  nn.CSubTable()({output, prev_h,}),}),})
  return _, next_h
end


function create_network()
    local x                  = nn.Identity()()
    local y                  = nn.Identity()()
    local prev_s             = nn.Identity()()
    local i                  = {[0] = nn.LookupTable(params.vocab_size,
                                                    params.rnn_size)(x)}
    local next_s             = {}
    local split              = {prev_s:split(2 * params.layers)}
    for layer_idx = 1, params.layers do
        local prev_c         = split[2 * layer_idx - 1]
        local prev_h         = split[2 * layer_idx]
        --print(prev_h)
	--print(i)
	--print(layer_idx)
	local dropped        = nn.Dropout(params.dropout)(i[layer_idx - 1])
	local next_c, next_h = lstm(dropped, prev_c, prev_h)
        if params.model_type == 'lstm' then
		local next_c, next_h = lstm(dropped, prev_c, prev_h)
	elseif params.model_type == 'gru' then
		local next_c, next_h = gru(dropped, prev_c, prev_h)
	else
		print "Model type not supported. Please input lstm or gru and try again."
	end
        table.insert(next_s, next_c)
        table.insert(next_s, next_h)
        i[layer_idx] = next_h
    end
    local h2y                = nn.Linear(params.rnn_size, params.vocab_size)
    local dropped            = nn.Dropout(params.dropout)(i[params.layers])
    local pred               = nn.LogSoftMax()(h2y(dropped))
    local err                = nn.ClassNLLCriterion()({pred, y})
    local module             = nn.gModule({x, y, prev_s},
                                      {err, nn.Identity()(next_s),pred})
    -- initialize weights
    module:getParameters():uniform(-params.init_weight, params.init_weight)
    return transfer_data(module)
end

--builds model using create_network
function setup()
    print("Creating " .. params.model_type .. " RNN network ..")
    local core_network = create_network()
    paramx, paramdx = core_network:getParameters()
    model.s = {}
    model.ds = {}
    model.start_s = {}
    for j = 0, params.seq_length do
        model.s[j] = {}
        for d = 1, 2 * params.layers do
            model.s[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
        end
    end
    for d = 1, 2 * params.layers do
        model.start_s[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
        model.ds[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size))
    end
    model.core_network = core_network
    model.rnns = g_cloneManyTimes(core_network, params.seq_length)
    model.norm_dw = 0
    model.err = transfer_data(torch.zeros(params.seq_length))
end

function reset_state(state)
    state.pos = 1
    if model ~= nil and model.start_s ~= nil then
        for d = 1, 2 * params.layers do
            model.start_s[d]:zero()
        end
    end
end

function reset_ds()
    for d = 1, #model.ds do
        model.ds[d]:zero()
    end
end

function fp(state)
    -- g_replace_table(from, to).  
    g_replace_table(model.s[0], model.start_s)
    
    -- reset state when we are done with one full epoch
    if state.pos + params.seq_length > state.data:size(1) then
        reset_state(state)
    end
    
    -- forward prop
    for i = 1, params.seq_length do
        local x = state.data[state.pos]
        local y = state.data[state.pos + 1]
        local s = model.s[i - 1]
        model.err[i], model.s[i] = unpack(model.rnns[i]:forward({x, y, s}))
        state.pos = state.pos + 1
    end
    
    -- next-forward-prop start state is current-forward-prop's last state
    g_replace_table(model.start_s, model.s[params.seq_length])
    
    -- cross entropy error
    return model.err:mean()
end

function bp(state)
    -- start on a clean slate. Backprop over time for params.seq_length.
    paramdx:zero()
    reset_ds()
    for i = params.seq_length, 1, -1 do
        -- to make the following code look almost like fp
        state.pos = state.pos - 1
        local x = state.data[state.pos]
        local y = state.data[state.pos + 1]
        local s = model.s[i - 1]
        -- Why 1?
        local derr = transfer_data(torch.ones(1))
	-- tmp stores the ds
	local out = transfer_data(torch.zeros(params.batch_size, params.vocab_size))
        local tmp = model.rnns[i]:backward({x, y, s},
                                           {derr, model.ds, out})[3]
	-- remember (to, from)
        g_replace_table(model.ds, tmp)
    end
    
    -- undo changes due to changing position in bp
    state.pos = state.pos + params.seq_length
    
    -- gradient clipping
    model.norm_dw = paramdx:norm()
    if model.norm_dw > params.max_grad_norm then
        local shrink_factor = params.max_grad_norm / model.norm_dw
        paramdx:mul(shrink_factor)
    end
    
    -- gradient descent step
    paramx:add(paramdx:mul(-params.lr))
end

--model trainer. Shifted earlier main.lua code into this.
function run_train()
	local states = {state_train, state_valid, state_test}
	for _, state in pairs(states) do
	    reset_state(state)
	end
	--setup()
	step = 0
	epoch = 0
	total_cases = 0
	--start with -1.
	local curr_perp = -1
	local best_perp = -1
	beginning_time = torch.tic()
	start_time = torch.tic()
	print("Starting training ..")
	words_per_step = params.seq_length * params.batch_size
	epoch_size = torch.floor(state_train.data:size(1) / params.seq_length)

	while epoch < params.max_max_epoch do

	    -- take one step forward
	    perp = fp(state_train)
	    if perps == nil then
		perps = torch.zeros(epoch_size):add(perp)
	    end
	    perps[step % epoch_size + 1] = perp
	    step = step + 1
	    
	    -- gradient over the step
	    bp(state_train)
	    
	    -- words_per_step covered in one step
	    total_cases = total_cases + params.seq_length * params.batch_size
	    epoch = step / epoch_size
	    
	    -- display details at some interval
	    if step % torch.round(epoch_size / 10) == 10 then
		wps = torch.floor(total_cases / torch.toc(start_time))
		since_beginning = g_d(torch.toc(beginning_time) / 60)
		print('epoch = ' .. g_f3(epoch) ..
		     ', train perp. = ' .. g_f3(torch.exp(perps:mean())) ..
		     ', wps = ' .. wps ..
		     ', dw:norm() = ' .. g_f3(model.norm_dw) ..
		     ', lr = ' ..  g_f3(params.lr) ..
		     ', since beginning = ' .. since_beginning .. ' mins.')
	    end
	    
	    -- run when epoch done
	    if step % epoch_size == 0 then
		curr_perp = run_valid()
		best_perp = save_model(best_perp,curr_perp)
		--if curr_perp<best_perp then
		--	save_model()
		--	best_perp = curr_perp
		--end
		if epoch > params.max_epoch then
		    params.lr = params.lr / params.decay
		end
	    end
	end
	curr_perp = run_valid()
--	if curr_perp<best_perp then
--		save_model()
--		best_perp = curr_perp
--	end
	save_model(best_perp,curr_perp)
	--run_test()
	print("Training is over ..")
	print("Best perplexity is: " .. best_perp)
end

function run_valid()
    print("Validating ..")
    -- again start with a clean slate
    reset_state(state_valid)
    
    -- no dropout in testing/validating
    g_disable_dropout(model.rnns)
    
    -- collect perplexity over the whole validation set
    local len = (state_valid.data:size(1) - 1) / (params.seq_length)
    local perp = 0
    for i = 1, len do
        perp = perp + fp(state_valid)
    end
    print("Validation set perplexity : " .. g_f3(torch.exp(perp / len)))
    g_enable_dropout(model.rnns)
    return torch.exp(perp/len) --returns probability
end


--not needed.
function run_test()
    print("Testing stage..")
    reset_state(state_test)
    g_disable_dropout(model.rnns)
    local perp = 0
    local len = state_test.data:size(1)
    
    -- no batching here
    g_replace_table(model.s[0], model.start_s)
    for i = 1, (len - 1) do
	xlua.progress(i,len-1)
        local x = state_test.data[i]
        local y = state_test.data[i + 1]
        perp_tmp, model.s[1] = unpack(model.rnns[1]:forward({x, y, model.s[0]}))
        perp = perp + perp_tmp[1]
        g_replace_table(model.s[0], model.s[1])
    end
    print("Test set perplexity : " .. g_f3(torch.exp(perp / (len - 1))))
    g_enable_dropout(model.rnns)
end

if gpu then
    g_init_gpu(arg)
end

--function saves model if we get a better perplexity at a stage.
function save_model(best_perp, curr_perp)
    local old_perp = -1
    if curr_perp < best_perp or best_perp==-1 then
	    if best_perp == -1 then
		    old_perp = "NA" --first model save
	    else
		    old_perp  = best_perp
	    end
	torch.save(params.model_dir .. params.model_name,model)
	print("Saved new best model")
    	best_perp = curr_perp
    end
    print("Current perplexity: " .. curr_perp .. ", Previous best: " .. old_perp)
    return best_perp
end

-- get data in batches
state_train = {data=transfer_data(ptb.traindataset(params.batch_size))}
state_valid =  {data=transfer_data(ptb.validdataset(params.batch_size))}
state_test =  {data=transfer_data(ptb.testdataset(params.batch_size))}

print("Network parameters:")
print(params)
print(params.layers)
print(params.dropout)
print("Setting model up for training..")
setup()
run_train()
print("Model is trained. Use result.lua to test.")



