eval.lin.mem = function(A, B, memory, x){
  "A gives result, B gives memory"
  v = rbind(memory,x)  
  result = A %*% v
  new.memory = B %*% v
  result  = list(result, new.memory)
  return(result)
}

gradients = function(A,B, mem.current, mem.prev, x.current, x.prev, y){
    # first let's try for grad A
    # Lets do a bunch of asserts about the sizes involved
    results = eval.lin.mem(A,B, mem.current, x.current)
    response = results[[1]]
    dy = y - response
    # what goes into the result
    v.response = rbind(mem.current, x.current)
    A.grad = dy %*% t(v.response)
    #now we try for grad B
    dm = (t(A) %*% dy)[1:nrow(mem.current),] #top m rows
    v.mem = rbind(mem.prev, x.prev)
    B.grad = dm %*% t(v.mem)
    result = list(A.grad, B.grad)
    return(result)
}

