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
cmd:option('-warm_start', '', 'model to start from')

-- Hyperparameters
cmd:option('-eta', 0.01, 'learning rate for SGD')
cmd:option('-max_epochs', 20, 'max epochs for NN training')

cmd:option('-backprop_length', 100, 'backprop length for RNN')
cmd:option('-batch_size', 32, 'batch size for RNN')
cmd:option('-hidden_size', 100, 'hidden size for RNN')
cmd:option('-eval_mode', 'greedy', 'evaluation mode for RNN')

function rnn_reshape(arr, backprop_length, batch_size)
  -- Reshape into a table of (N / batch_size) x backprop_length tensors
  local N = arr:size(1)
  -- In case batch_size doesn't divide N
  arr = arr:narrow(1, 1, math.floor(N/batch_size) * batch_size)
  return arr:reshape(batch_size, math.floor(N/batch_size)):split(backprop_length, 2)
end

function LSTM_model(hidden_size, vocab_size, embed_dim)
   if opt.warm_start ~= '' then
     return torch.load(opt.warm_start).model
   end

   local model = nn.Sequential()
   model:add(nn.LookupTable(vocab_size, embed_dim))
   model:add(nn.SplitTable(1, 3))

   local stepmodule = nn.Sequential()

   local rnn = nn.Sequential()
   rnn:add(nn.LSTM(embed_dim, hidden_size))
   stepmodule:add(rnn)

   local output = nn.Sequential()
   output:add(nn.Linear(hidden_size, 2))
   output:add(nn.LogSoftMax())
   stepmodule:add(output)

   model:add(nn.Sequencer(stepmodule))
   model:remember('both')

   return model
end

-- Currently broken
function model_eval(model, criterion, X, Y, X_answers)
  -- batch evaluation
  model:evaluate()
  if opt.eval_mode == 'greedy' then
    -- join and flatten out the inputs
    X_arr = nn.JoinTable(2):forward(X)
    Y_arr = nn.JoinTable(2):forward(Y)
    X_arr = nn.Reshape(X_arr:size(1)*X_arr:size(2)):forward(X_arr)
    Y_arr = nn.Reshape(Y_arr:size(1)*Y_arr:size(2)):forward(Y_arr)

    -- array to store output, which we will join into a tensor later
    local output_arr = {}

    local total_correct = 0
    local idx = 1
    local next_char = X[idx]
    while idx <= X:size(1) do

      -- feed element into RNN
      local output = model:forward(next_char)
      if output == Y[idx] then
        total_correct = total_correct + 1
      end

      -- increment index
      idx = idx + 1

      -- if output is space feed in a space
      if output == 2 then
         next_char = space 
      else
         next_char = X[idx]
      end
      local X_batch = X[i]:split(1, 2) 
      local Y_batch = Y[i]:split(1, 2)
      local outputs = model:forward(X_batch)

    local loss = criterion:forward(outputs, Y_batch)
    total_loss = total_loss + loss * batch_size
  end

  end
  
  return total_loss / N
end

function train_LSTM_model(X, Y, valid_X, valid_Y, vocab_size, embed_dim)
   local eta = opt.eta
   local max_epochs = opt.max_epochs
   local batch_size = opt.batch_size
   local backprop_length = opt.backprop_length
   local hidden_size = opt.hidden_size

   local model = LSTM_model(hidden_size, vocab_size, embed_dim)
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
      local X_batch = X[i]:t()
      local Y_batch = Y[i]:t()

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

        -- renormalize gradients with max norm 5
        local max_grad_norm = math.abs(grads:max())
        grads:mul(5):div(max_grad_norm)
        print('Gradient Norm:', grads:norm(2))

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
   local X = f:read('train_input'):all()
   local Y = f:read('train_output'):all()
   local valid_X = f:read('valid_input'):all()
   local valid_Y = f:read('valid_output'):all()
   local test_X = f:read('test_input'):all()
   local vocab_size = f:read('V'):all()[1]

   local backprop_length = opt.backprop_length
   local batch_size = opt.batch_size
   local embed_dim = 6 -- change this??

   -- Train.
   if opt.classifier == 'lstm' then
     X = rnn_reshape(X, backprop_length, batch_size)
     Y = rnn_reshape(Y, backprop_length, batch_size)
     valid_X = rnn_reshape(valid_X, backprop_length, batch_size)
     valid_Y = rnn_reshape(valid_Y, backprop_length, batch_size)
     test_X = rnn_reshape(test_X, backprop_length, batch_size)

     local hidden_size = opt.hidden_size
     local max_epochs = opt.max_epochs

     local model = train_LSTM_model(X, Y, valid_X, valid_Y, vocab_size, 6)
   end

   -- Test.
end

main()
