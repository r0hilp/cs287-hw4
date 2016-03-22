-- Only requirement allowed
require("hdf5")
require("optim")
require("nn")
require("rnn")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-classifier', 'lstm', 'classifier to use')
cmd:option('-model_out_name', 'PTB', 'model output name to use')

-- Hyperparameters
cmd:option('-eta', 0.01, 'learning rate for SGD')
cmd:option('-max_epochs', 20, 'max epochs for NN training')

cmd:option('-backprop_length', 100, 'backprop length for RNN')
cmd:option('-batch_size', 300, 'batch size for RNN')
cmd:option('-hidden_size', 100, 'hidden size for RNN')

function rnn_reshape(arr, backprop_length, batch_size)
  -- Reshape into a table of (N / batch_size) x backprop_length tensors
  local N = arr:size(1)
  -- In case batch_size doesn't divide N
  arr = arr:narrow(1, 1, math.floor(N/batch_size) * batch_size)
  return arr:reshape(batch_size, math.floor(N/batch_size)):split(backprop_length, 2)
end

function LSTM_model(input_size, hidden_size)

   local model = nn.Sequential()

   local stepmodel = nn.Sequential()

   local rnn = nn.Sequential()
   rnn:add(nn.FastLSTM(1, hidden_size))
   stepmodel:add(rnn)

   local output = nn.Sequential()
   output:add(nn.Linear(hidden_size, 2))
   output:add(nn.LogSoftMax())
   stepmodel:add(output)

   model:add(nn.Sequencer(stepmodel))
   model:remember('both')

   return model
end

function train_LSTM_model(X, Y, valid_X, valid_Y)
   local eta = opt.eta
   local max_epochs = opt.max_epochs
   local batch_size = opt.batch_size
   local backprop_length = opt.backprop_length
   local hidden_size = opt.hidden_size

   local model = LSTM_model(backprop_length, hidden_size)
   local criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

   -- call once
   local params, grads = model:getParameters()
   
   -- initialize params to uniform between -0.05, 0.05
   params = torch.rand(params:size()):div(10):csub(0.05)

   -- sgd state
   local state = { learningRate = eta }

   local total_loss = 0

   model:training()
   for i = 1, #X do
      print('Batch Number:', i, 'of', #X)
      print(X[i]:size())
      local X_batch = X[i]:split(1, 2)
      local Y_batch = Y[i]:split(1, 2)

      -- process Y_batch
      for j = 1, #Y_batch do
        Y_batch[j] = Y_batch[j]:squeeze()
      end

      -- closure to return err, df/dx
      local func = function(x)
        -- get new parameters
        if x ~= params then
          params:copy(x)
        end
        -- reset gradients
        grads:zero()

        -- forward step
        local inputs = X_batch
        local outputs = model:forward(inputs)
        local loss = criterion:forward(outputs, Y_batch)

        -- track errors
        total_loss = total_loss + loss * batch_size

        -- compute gradients
        local df_dz = criterion:backward(outputs, Y_batch)
        model:backward(inputs, df_dz)

        -- renormalize gradients
        -- print(grads)
        -- grads:renorm(2, 1, 5)
        print(grads:norm(2))

        return loss, grads
      end

      optim.sgd(func, params, state)
      torch.save(opt.model_out_name .. '_' .. opt.classifier .. '.t7', { model = model })
   end

   return model
end

function main() 
   -- Parse input params
   opt = cmd:parse(arg)
   local f = hdf5.open(opt.datafile, 'r')
   local X_arr = f:read('train_input'):all():double()
   local Y_arr = f:read('train_output'):all()
   local valid_X_arr = f:read('valid_input'):all()
   local valid_Y_arr = f:read('valid_output'):all()
   local test_X_arr = f:read('test_input'):all()

   local backprop_length = opt.backprop_length
   local batch_size = opt.batch_size

   -- Train.
   if opt.classifier == 'lstm' then
     local X = rnn_reshape(X_arr, backprop_length, batch_size)
     local Y = rnn_reshape(Y_arr, backprop_length, batch_size)
     local valid_X = rnn_reshape(valid_X_arr, backprop_length, batch_size)
     local valid_Y = rnn_reshape(valid_Y_arr, backprop_length, batch_size)
     local test_X = rnn_reshape(test_X_arr, backprop_length, batch_size)

     local hidden_size = opt.hidden_size
     local max_epochs = opt.max_epochs
     -- local model = LSTM_model(backprop_length, hidden_size)
     local model = train_LSTM_model(X, Y)
   end

   -- Test.
end

main()
