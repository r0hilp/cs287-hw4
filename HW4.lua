-- Only requirement allowed
require("hdf5")
require("optim")
require("nn")
require("rnn")

cmd = torch.CmdLine()

-- Cmd Args
cmd:option('-datafile', '', 'data file')
cmd:option('-classifier', 'lstm', 'classifier to use')
cmd:option('-kaggle_answers', 'data/valid_chars_kaggle_answer.txt', 'validation kaggle answers')

cmd:option('-warm_start', '', 'torch file with previous model')
cmd:option('-action', 'train', 'action: train or test')
cmd:option('-model_out_name', 'train', 'output file name of model')
cmd:option('-debug', 0, 'print training debug')

-- Hyperparameters
cmd:option('-alpha', 0.1, 'smoothing alpha')
cmd:option('-gram_size', 5, 'size of context')
cmd:option('-test_method', 'greedy', 'greedy or dp')

cmd:option('-eta', 0.01, 'learning rate for SGD')
cmd:option('-min_epochs', 5, 'min epochs for NN training')
cmd:option('-max_epochs', 20, 'max epochs for NN training')
cmd:option('-nnlm_batch_size', 32, 'batch size for nnlm')

cmd:option('-L2s', 1, 'normalize L2 of word embeddings')
cmd:option('-embed', 15, 'size of word embeddings')
cmd:option('-hidden', 100, 'size of hidden layer for neural network')

cmd:option('-max_grad', 5, 'maximum gradient for renormalization')
cmd:option('-backprop_length', 50, 'backprop length for RNN')
cmd:option('-batch_size', 32, 'batch size for RNN')
cmd:option('-hidden_size', 100, 'hidden size for RNN')
cmd:option('-space_cutoff', 0.6, 'cutoff probability for a space in RNN greedy search')

function compute_mse(kaggle_answers, spaces_per_sentence)
   -- read kaggle answers into a table (put into rnn_model_kaggle?)
   local valid_kaggle_answers = {}
   local mse = 0
   if kaggle_answers ~= '' then
     local f = io.open(opt.kaggle_answers)
     -- skip header
     f:read("*l")
     local count = 1
     local line = f:read("*l")
     while line ~= nil do
       valid_kaggle_answers[count] = tonumber(line:split(",")[2])
       line = f:read("*l")
       count = count + 1
     end
     for i, j in pairs(valid_kaggle_answers) do
       mse = mse + (j - spaces_per_sentence[i]) * (j - spaces_per_sentence[i])
     end
     mse = mse / #spaces_per_sentence
     print('Valid MSE:', mse)
   end
   return mse
end

function greedy_search(X, CM, model)
  local gram_size = opt.gram_size
  local spaces = {}
  local cur_sent = 1
  spaces[1] = 0

  local context
  local is_space = false
  local i = 1
  while i <= X:size(1) - gram_size + 1 do
    -- Get context
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

    if context[context:size(1)] == sentence_char then
      cur_sent = cur_sent + 1
      spaces[cur_sent] = 0
    end

    -- Evaluate
    if CM then
      local h = hash(context)
      if CM[h] and CM[h][2] then
        if CM[h][1] == nil or CM[h][2] > CM[h][1] then
          -- Predict space
          is_space = true
          spaces[cur_sent] = spaces[cur_sent] + 1
        end
      end
    elseif model then
      local pred = model:forward(context)
      if pred[2] > pred[1] then
        is_space = true
        spaces[cur_sent] = spaces[cur_sent] + 1
      end
    end
  end

  return spaces
end

function dp_search(X, CM, model)
  local gram_size = opt.gram_size

  -- n gram dp, bitmask for spaces after w_{i-n+2},...,w_{i-1}
  -- to get p(space | w_i, ..., w_{i-n+2})
  local pi = torch.Tensor(X:size(1), 2 ^ (gram_size - 2)):fill(-math.huge)
  local bp = torch.zeros(X:size(1), 2 ^ (gram_size - 2)):long()
  local cache = {}

  -- initialize
  pi[gram_size-2]:zero()

  print(X:size())
  for i = gram_size-1, X:size(1) do
    if i % 10000 == 0 then print(i) end
    for k = 0, 2 ^ (gram_size - 2) - 1 do
      for c = 1, 2 do
        local prev_k = math.floor(k/2) + (c-1) * (2 ^ (gram_size - 3))
        local chars = X:narrow(1, i-gram_size+2, gram_size - 1)
        local context = {}
        for j = 1, gram_size - 2 do
          table.insert(context, chars[j])
          if bit.band(prev_k, 2 ^ (gram_size-2-j)) > 0 then
            table.insert(context, space_char)
          end
        end
        table.insert(context, chars[gram_size-1])
        context = torch.LongTensor(context)
        context = context:narrow(1, context:size(1) - gram_size + 2, gram_size - 1)

        local z
        if CM then
          local h = hash(context)
          if cache[h] then
            --print('cached', h)
            z = cache[h]
          else
            z = torch.Tensor{0.6, 0.4}
            if CM[h] then
              if CM[h][1] then
                z[1] = CM[h][1]
              else
                z[1] = opt.alpha
              end
              if CM[h][2] then
                z[2] = CM[h][2]
              else
                z[2] = opt.alpha
              end
            end
            z:div(z:sum())
            z:log()
            cache[h] = z:clone()
          end
        elseif model then
          local h = hash(context)
          if cache[h] then
            z = cache[h]
          else
            z = model:forward(context)
            cache[h] = z:clone()
          end
        end

        --print('i,k,c:')
        --print(i, k, c)
        --print(context, z)
        --io.read()

        local cur = (k%2) + 1
        if pi[i-1][prev_k + 1] + z[cur] > pi[i][k + 1] then
          pi[i][k+1] = pi[i-1][prev_k + 1] + z[cur]
          bp[i][k+1] = prev_k -- annoying indices
        end
      end
    end
  end

  --print(pi)

  local spaces = {}
  local _, p = torch.max(pi[X:size(1)], 1)
  p = p[1] - 1

  for i = X:size(1), gram_size - 2, -1 do
    if p%2 > 0 then
      table.insert(spaces, i)
    end
    p = bp[i][p + 1]
    --io.write(p, " ")
  end

  local cur_sent = 1
  local space_sent = {}
  space_sent[1] = 0

  local i = #spaces
  for j = 1, X:size(1) do
    if X[j] == sentence_char then
      cur_sent = cur_sent + 1
      space_sent[cur_sent] = 0
    end

    if spaces[i] == j then
      space_sent[cur_sent] = space_sent[cur_sent] + 1
      i = i - 1
    end
  end

  return space_sent
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
  model:add(nn.Linear(opt.hidden, 2))
  model:add(nn.LogSoftMax())

  return model
end

function eval_nnlm(model, criterion, X, Y)
    -- batch eval
    model:evaluate()
    local N = X:size(1)
    local batch_size = opt.batch_size

    local total_loss = 0
    local total_correct = 0
    for batch = 1, X:size(1), batch_size do
        local sz = batch_size
        if batch + batch_size > N then
          sz = N - batch + 1
        end
        local X_batch = X:narrow(1, batch, sz)
        local Y_batch = Y:narrow(1, batch, sz)

        local outputs = model:forward(X_batch)
        local loss = criterion:forward(outputs, Y_batch)
        total_loss = total_loss + loss * sz
        -- unfortunate pathology
        local correct
        if outputs:size():size() == 1 then
          local _, argmax = outputs:max(1)
          if argmax == Y_batch[1] then
            correct = 1
          else
            correct = 0
          end
        else
          local _, argmax = outputs:max(2)
          correct = argmax:squeeze():eq(Y_batch):sum()
        end
        total_correct = total_correct + correct
    end

    return total_loss / N, total_correct / N
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
            -- unfortunate pathology
            local correct
            if outputs:size():size() == 1 then
              local _, argmax = outputs:max(1)
              if argmax == Y_batch[1] then
                correct = 1
              else
                correct = 0
              end
            else
              local _, argmax = outputs:max(2)
              correct = argmax:squeeze():eq(Y_batch):sum()
            end
            total_correct = total_correct + correct

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
      print('Train percent:', total_correct / N)

      local loss, percent = eval_nnlm(model, criterion, valid_X, valid_Y)
      print('Valid perplexity:', torch.exp(loss))
      print('Valid percent:', percent)

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
  local ret = arr:reshape(batch_size, math.floor(N/batch_size)):split(backprop_length, 2)
  -- transpose precompute
  for i,_ in ipairs(ret) do
    ret[i] = ret[i]:t()
  end
  return ret
end

function RNN_model(hidden_size, vocab_size, embed_dim)
  if opt.warm_start ~= '' then
    return torch.load(opt.warm_start).model
  end

  local model = nn.Sequential()
  return model
end

function GRU_model(hidden_size, vocab_size, embed_dim, dropout)
   dropout = dropout or 0

   if opt.warm_start ~= '' then
     return torch.load(opt.warm_start).model
   end

   local model = nn.Sequential()
   model:add(nn.LookupTable(vocab_size, embed_dim))
   model:add(nn.SplitTable(1, 3))

   local stepmodule = nn.Sequential()

   local rnn = nn.Sequential()
   rnn:add(nn.GRU(embed_dim, hidden_size, nil, dropout))
   stepmodule:add(rnn)

   local output = nn.Sequential()
   output:add(nn.Linear(hidden_size, 2))
   output:add(nn.LogSoftMax())
   stepmodule:add(output)

   model:add(nn.Sequencer(stepmodule))
   model:remember('both')

   return model
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
   rnn:add(nn.FastLSTM(embed_dim, hidden_size))
   stepmodule:add(rnn)

   local output = nn.Sequential()
   output:add(nn.Linear(hidden_size, 2))
   output:add(nn.LogSoftMax())
   stepmodule:add(output)

   model:add(nn.Sequencer(stepmodule))
   model:remember('both')

   return model
end

function rnn_model_eval(model, criterion, X, Y)
  -- batch evaluate
  model:evaluate()
  local N = (#X - 1) * opt.backprop_length * opt.batch_size + X[#X]:size(1) * opt.batch_size

  local total_loss = 0
  local total_correct = 0
  for i = 1, #X do
      local X_batch = X[i]
      local Y_batch = Y[i]
      local sz = X_batch:size(2)

      local outputs = model:forward(X_batch)
      local loss = criterion:forward(outputs, Y_batch)
      total_loss = total_loss + loss * sz
      local _, argmax = nn.JoinTable(1):forward(outputs):max(2)
      local correct = argmax:squeeze():eq(Y_batch):sum()
      total_correct = total_correct + correct
  end

  return total_loss / N, total_correct / N
end

function rnn_model_kaggle(model, kaggle_X_arr, space_char, sentence_char, space_cutoff)

  model:evaluate()

  -- get spaces per sentence
  local spaces_per_sentence = {}
  spaces_per_sentence[1] = 0
  local curr_sentence = 1
  for i = 1, kaggle_X_arr:size(1)-1 do
    local curr_char = kaggle_X_arr[i]
    -- update sentence spaces
    if curr_char == sentence_char then
      curr_sentence = curr_sentence + 1
      spaces_per_sentence[curr_sentence] = 0
    end
    -- feed curr_char into RNN
    local out = nn.JoinTable(1):forward(model:forward(torch.Tensor({curr_char})))
    -- local max, argmax = out:max(1)
    -- feed any spaces obtained into RNN and update spaces_per_sentence
    while math.exp(out[2]) >= space_cutoff do
      out = nn.JoinTable(1):forward(model:forward(torch.Tensor({space_char})))
      -- local new_max, new_argmax = out:max(1)
      -- max = new_max
      -- argmax = new_argmax
      spaces_per_sentence[curr_sentence] = spaces_per_sentence[curr_sentence] + 1
    end
  end

  return spaces_per_sentence
end

function train_RNN_model(X, Y, valid_X, valid_Y, vocab_size, embed_dim, space_char)
   local eta = opt.eta
   local max_epochs = opt.max_epochs
   local backprop_length = opt.backprop_length
   local hidden_size = opt.hidden_size
   local N = (#X - 1) * backprop_length * opt.batch_size + X[#X]:size(1) * opt.batch_size

   local model
   if opt.classifier == 'lstm' then
     model = LSTM_model(hidden_size, vocab_size, embed_dim)
   elseif opt.classifier == 'gru' then
     model = GRU_model(hidden_size, vocab_size, embed_dim) 
   end
   local criterion = nn.SequencerCriterion(nn.ClassNLLCriterion())

   -- call once
   local params, grads = model:getParameters()
   
   if opt.warm_start == '' then
     -- initialize params to uniform between -0.05, 0.05
     params:uniform(-0.05, 0.05)
   end

   local prev_loss = 1e10
   local epoch = 1
   local timer = torch.Timer()
   while epoch <= max_epochs do
     print('Epoch:', epoch)
     local epoch_time = timer:time().real
     local total_loss = 0
     local total_correct = 0

     -- sgd state
     local state = { learningRate = eta }

     model:training()
     for i = 1, #X do
        --print('Batch Number:', i, 'of', #X)
        local X_batch = X[i]
        local Y_batch = Y[i]
        local sz = X_batch:size(2)

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
          total_loss = total_loss + loss * sz
          local _, argmax = nn.JoinTable(1):forward(outputs):max(2)
          local correct = argmax:squeeze():eq(Y_batch):sum()
          total_correct = total_correct + correct

          -- compute gradients
          local df_do = criterion:backward(outputs, Y_batch)
          model:backward(inputs, df_do)

          -- renormalize gradients with norm too big 
          local max_grad_norm = math.abs(grads:max())
          if max_grad_norm > opt.max_grad then
            grads:mul(opt.max_grad):div(max_grad_norm)
          end

          return loss, grads
        end

        optim.sgd(func, params, state)
      end

      print('Train perplexity:', torch.exp(total_loss / N))
      print('Train percent:', total_correct / N)

      local loss, percent = rnn_model_eval(model, criterion, valid_X, valid_Y)
      print('Valid perplexity:', torch.exp(loss))
      print('Valid percent:', percent)

      print('time for one epoch: ', (timer:time().real - epoch_time) * 1000, 'ms')
      print('')

      -- forget to restart sequence
      model:forget()
      if loss > prev_loss and epoch > opt.min_epochs then
        -- halve learning rate
        eta = eta / 2
      end
      prev_loss = loss
      epoch = epoch + 1
      torch.save(opt.model_out_name .. '_' .. opt.classifier .. '.t7', { model = model })
    end
    print('Trained', epoch, 'epochs')

   return model
end

function main() 
   -- Parse input params
   opt = cmd:parse(arg)
   local f = hdf5.open(opt.datafile, 'r')

   local X_arr = f:read('train_input'):all():long()
   local Y_arr = f:read('train_output'):all():long()
   local valid_X_arr = f:read('valid_input'):all():long()
   local valid_Y_arr = f:read('valid_output'):all():long()
   local valid_kaggle_X_arr = f:read('valid_kaggle_input'):all():long()
   local test_X_arr = f:read('test_input'):all():long()
   vocab_size = f:read('vocab_size'):all():long()[1]
   space_char = f:read('space_char'):all():long()[1]
   sentence_char = f:read('sentence_char'):all():long()[1]

   local backprop_length = opt.backprop_length
   local batch_size = opt.batch_size
   local gram_size = opt.gram_size
   local embed_dim = opt.embed

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
     local spaces
     local valid_spaces
     if opt.test_method == 'greedy' then
       valid_spaces = greedy_search(valid_kaggle_X_arr, CM, nil) 
       spaces = greedy_search(test_X_arr, CM, nil) 
     elseif opt.test_method == 'dp' then
       valid_spaces = dp_search(valid_kaggle_X_arr, CM, nil) 
       spaces = dp_search(test_X_arr, CM, nil)
     end
     compute_mse(opt.kaggle_answers, valid_spaces)

     -- Write spaces
     local space_f = io.open('test_spaces_count.txt', 'w')
     space_f:write("ID,Count")
     for i,v in ipairs(spaces) do
       space_f:write(i .. "," .. v .. "\n")
     end

   elseif opt.classifier == 'nnlm' then
     local X = X_to_context(X_arr, gram_size)
     local valid_X = X_to_context(valid_X_arr, gram_size)
     local Y = Y_arr:narrow(1, gram_size-1, Y_arr:size(1) - gram_size + 2)
     local valid_Y = valid_Y_arr:narrow(1, gram_size-1, valid_Y_arr:size(1) - gram_size + 2)

     local model
     if opt.action == 'train' then
       model, _ = train_nnlm(X, Y, valid_X, valid_Y)
     elseif opt.action == 'test' then
       model = torch.load(opt.warm_start).model
     end

     -- Test
     local spaces
     local valid_spaces
     if opt.test_method == 'greedy' then
       valid_spaces = greedy_search(valid_kaggle_X_arr, nil, model)
       spaces = greedy_search(test_X_arr, nil, model) 
     elseif opt.test_method == 'dp' then
       valid_spaces = dp_search(valid_kaggle_X_arr, nil, model)
       spaces = dp_search(test_X_arr, nil, model)
     end
     compute_mse(opt.kaggle_answers, valid_spaces)

     -- Write spaces
     local space_f = io.open('test_spaces_count_nnlm.txt', 'w')
     space_f:write("ID,Count")
     for i,v in ipairs(spaces) do
       space_f:write(i .. "," .. v .. "\n")
     end

   elseif opt.classifier == 'lstm' or opt.classifier == 'gru' then
     local X = rnn_reshape(X_arr, backprop_length, batch_size)
     local Y = rnn_reshape(Y_arr, backprop_length, batch_size)
     local valid_X = rnn_reshape(valid_X_arr, backprop_length, batch_size)
     local valid_Y = rnn_reshape(valid_Y_arr, backprop_length, batch_size)
     local test_X = rnn_reshape(test_X_arr, backprop_length, batch_size)

     local hidden_size = opt.hidden_size
     local max_epochs = opt.max_epochs

     local model = train_RNN_model(X, Y, valid_X, valid_Y, vocab_size, opt.embed, space_char, sentence_char)

     local spaces_per_sentence = rnn_model_kaggle(model, valid_kaggle_X_arr, space_char, sentence_char, opt.space_cutoff)
     compute_mse(opt.kaggle_answers, spaces_per_sentence)

     -- Test
     local test_spaces_per_sentence = rnn_model_kaggle(model, test_X_arr, space_char, sentence_char, opt.space_cutoff)
     local f = io.open(opt.model_out_name .. '.preds', 'w')
     f:write('ID,Count\n')     
     for i, j in pairs(test_spaces_per_sentence) do
       f:write(i..','..j..'\n')
     end
   end

end

main()
