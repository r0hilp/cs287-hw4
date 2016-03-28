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

cmd:option('-warm_start', '', 'torch file with previous model')
cmd:option('-model_out_name', 'train', 'output file name of model')
cmd:option('-debug', 0, 'print training debug')

-- Hyperparameters
cmd:option('-alpha', 0.1, 'smoothing alpha')
cmd:option('-gram_size', 2, 'size of context')

cmd:option('-eta', 0.01, 'learning rate for SGD')
cmd:option('-min_epochs', 5, 'min epochs for NN training')
cmd:option('-max_epochs', 20, 'max epochs for NN training')
cmd:option('-nnlm_batch_size', 32, 'batch size for nnlm')

cmd:option('-L2s', 1, 'normalize L2 of word embeddings')
cmd:option('-embed', 15, 'size of word embeddings')
cmd:option('-hidden', 100, 'size of hidden layer for neural network')

cmd:option('-backprop_length', 100, 'backprop length for RNN')
cmd:option('-batch_size', 32, 'batch size for RNN')
cmd:option('-hidden_size', 100, 'hidden size for RNN')
cmd:option('-eval_mode', 'greedy', 'evaluation mode for RNN')

function greedy_search(X, CM, model)
  local gram_size = opt.gram_size
  local spaces = {}

  local context
  local is_space = false
  local i = 1
  while i <= X:size(1) - gram_size + 1 do
    if i == 1 then
      context = X:narrow(1, 1, gram_size - 1)
      i = gram_size
    else
      if is_space then
        -- Last predicted space
        context = torch.cat(context:narrow(1, 2, context:size(1) - 1), torch.LongTensor{space_char}) 
        is_space = false
      else
        context = torch.cat(context:narrow(1, 2, context:size(1) - 1), torch.LongTensor{X[i]}) 
        i = i + 1
      end
    end

    -- Evaluate
    if CM then
      local h = hash(context)
      if CM[h] and CM[h][2] then
        if CM[h][1] == nil or CM[h][2] > CM[h][1] then
          -- Predict space
          is_space = true
          table.insert(spaces, i)
        end
      end
    elseif model then
      local pred = model:forward(context)
      if pred[2] > pred[1] then
        is_space = true
        table.insert(spaces, i)
      end
    end
  end

  return spaces
end

function dp_search(X, CM, model)

end

function X_to_context(X, gram_size)
  local context = torch.Tensor(X:size(1) - gram_size + 2, gram_size-1)
  for i = gram_size - 1, X:size(1) do
    context[i-gram_size+2] = X:narrow(1, i-gram_size+2, gram_size-1)
  end
  return context
end

function hash(context)
  -- Hashes ngram context for 
  local total = 0
  for i = 1, context:size(1) do
    total = total + (context[i] - 1) * (vocab_size ^ (context:size(1)-i))
  end
  return total
end

function make_count_matrix(X, Y, gram_size)
  -- Construct count matrix
  local CM = {}
  for i = 1, X:size(1) do
    local h = hash(X[i])
    local is_space = Y[i]
    if CM[h] == nil then
      CM[h] = {}
      CM[h][is_space] = 1
    else
      if CM[h][is_space] == nil then
        CM[h][is_space] = 1
      else
        CM[h][is_space] = CM[h][is_space] + 1
      end
    end
  end

  return CM
end

function eval_count_model(X, Y, CM, gram_size)
  local alpha = opt.alpha

  local preds = torch.Tensor(Y:size(1), 2):fill(alpha)
  for i = 1, X:size(1) do
    local h = hash(X[i])
    for j = 1, 2 do
      if CM[h] and CM[h][j] then
        preds[i][j] = preds[i][j] + CM[h][j]
      end
    end
  end
  preds:cdiv(preds:sum(2):expand(preds:size(1), 2))

  return preds
end

function NNLM()
  if opt.warm_start ~= '' then
    return torch.load(opt.warm_start).model
  end

  local window_size = opt.gram_size - 1

  local model = nn.Sequential()
  model:add(nn.LookupTable(vocab_size, opt.embed))
  model:add(nn.View(opt.embed * window_size)) -- concat

  model:add(nn.Linear(opt.embed * window_size, opt.hidden))
  model:add(nn.Tanh())
  model:add(nn.Linear(opt.hidden, vocab_size))
  model:add(nn.LogSoftMax())

  return model
end

function model_eval(model, criterion, X, Y)
    -- batch eval
    model:evaluate()
    local N = X:size(1)
    local batch_size = opt.batch_size

    local total_loss = 0
    for batch = 1, X:size(1), batch_size do
        local sz = batch_size
        if batch + batch_size > N then
          sz = N - batch + 1
        end
        local X_batch = X:narrow(1, batch, sz)
        local Y_batch = Y:narrow(1, batch, sz)

        local outputs = model:forward(X_batch)
        local loss = criterion:forward(outputs, Y_batch)
        total_loss = total_loss + loss * batch_size
    end

    return total_loss / N
end

function train_nnlm(X, Y, valid_X, valid_Y)
  local eta = opt.eta
  local batch_size = opt.nnlm_batch_size
  local max_epochs = opt.max_epochs
  local N = X:size(1)

  local model = NNLM()
  local criterion = nn.ClassNLLCriterion()

  -- only call this once
  local params, grads = model:getParameters()

  -- sgd state
  local state = { learningRate = eta }

  local prev_loss = 1e10
  local epoch = 1
  local timer = torch.Timer()
  while epoch <= max_epochs do
      print('Epoch:', epoch)
      local epoch_time = timer:time().real
      local total_loss = 0
      local total_correct = 0

      -- shuffle for batches
      local shuffle = torch.randperm(N):long()
      X = X:index(1, shuffle)
      Y = Y:index(1, shuffle)

      -- loop through each batch
      model:training()
      for batch = 1, N, batch_size do
          if opt.debug == 1 then
            if ((batch - 1) / batch_size) % 300 == 0 then
              print('Sample:', batch)
              print('Current train loss:', total_loss / batch)
              print('Current time:', 1000 * (timer:time().real - epoch_time), 'ms')
            end
          end
          local sz = batch_size
          if batch + batch_size > N then
            sz = N - batch + 1
          end
          local X_batch = X:narrow(1, batch, sz)
          local Y_batch = Y:narrow(1, batch, sz)

          -- closure to return err, df/dx
          local func = function(x)
            -- get new parameters
            if x ~= params then
              params:copy(x)
            end
            -- reset gradients
            grads:zero()

            -- forward
            local inputs = X_batch
            local outputs = model:forward(inputs)
            local loss = criterion:forward(outputs, Y_batch)

            -- track errors
            total_loss = total_loss + loss * batch_size

            -- compute gradients
            local df_do = criterion:backward(outputs, Y_batch)
            model:backward(inputs, df_do)

            return loss, grads
          end

          optim.sgd(func, params, state)

          -- normalize weights
          --if opt.L2s > 0 then
            --local w = model:get(1).weight
            --w:renorm(2, 2, opt.L2s)
          --end
      end

      print('Train perplexity:', torch.exp(total_loss / N))

      local loss = model_eval(model, criterion, valid_X, valid_Y)
      print('Valid perplexity:', torch.exp(loss))

      print('time for one epoch: ', (timer:time().real - epoch_time) * 1000, 'ms')
      print('')
      if loss > prev_loss and epoch > opt.min_epochs then
        prev_loss = loss
        break
      end
      prev_loss = loss
      epoch = epoch + 1
      torch.save(opt.model_out_name .. '_' .. opt.classifier .. '.t7', { model = model })
  end
  print('Trained', epoch, 'epochs')
  return model, prev_loss
end

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
<<<<<<< HEAD
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
   local X_arr = f:read('train_input'):all():long()
   local Y_arr = f:read('train_output'):all():long()
   local valid_X_arr = f:read('valid_input'):all():long()
   local valid_Y_arr = f:read('valid_output'):all():long()
   local test_X_arr = f:read('test_input'):all():long()
   vocab_size = f:read('vocab_size'):all():long()[1]
   space_char = f:read('space_char'):all():long()[1]

   local backprop_length = opt.backprop_length
   local batch_size = opt.batch_size
   local gram_size = opt.gram_size

   -- Train.
   if opt.classifier == 'count' then
     local X = X_to_context(X_arr, gram_size)
     local valid_X = X_to_context(valid_X_arr, gram_size)
     local Y = Y_arr:narrow(1, gram_size-1, Y_arr:size(1) - gram_size + 2)
     local valid_Y = valid_Y_arr:narrow(1, gram_size-1, valid_Y_arr:size(1) - gram_size + 2)

     local CM = make_count_matrix(X, Y, gram_size)

     local preds = eval_count_model(X, Y, CM, gram_size)
     local valid_preds = eval_count_model(valid_X, valid_Y, CM, gram_size)

     local train_nll = nn.ClassNLLCriterion():forward(preds:log(), Y)
     local valid_nll = nn.ClassNLLCriterion():forward(valid_preds:log(), valid_Y)
     print('Train perplexity:', torch.exp(train_nll))
     print('Valid perplexity:', torch.exp(valid_nll))

     -- Test
     local spaces = greedy_search(test_X_arr, CM, nil)
     -- Write space locations
     local space_f = io.open('test_spaces_count.txt', 'w')
     for i,v in ipairs(spaces) do
       space_f:write(v .. "\n")
     end

   elseif opt.classifier == 'nnlm' then
     local X = X_to_context(X_arr, gram_size)
     local valid_X = X_to_context(valid_X_arr, gram_size)
     local Y = Y_arr:narrow(1, gram_size-1, Y_arr:size(1) - gram_size + 2)
     local valid_Y = valid_Y_arr:narrow(1, gram_size-1, valid_Y_arr:size(1) - gram_size + 2)

     local model, _ = train_nnlm(X, Y, valid_X, valid_Y)

   elseif opt.classifier == 'lstm' then
     local X = rnn_reshape(X_arr, backprop_length, batch_size)
     local Y = rnn_reshape(Y_arr, backprop_length, batch_size)
     local valid_X = rnn_reshape(valid_X_arr, backprop_length, batch_size)
     local valid_Y = rnn_reshape(valid_Y_arr, backprop_length, batch_size)
     local test_X = rnn_reshape(test_X_arr, backprop_length, batch_size)

     local hidden_size = opt.hidden_size
     local max_epochs = opt.max_epochs

     local model = train_LSTM_model(X, Y, valid_X, valid_Y, vocab_size, 6)
   end

end

main()
