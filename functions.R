eval.lin.mem = function(A, B, memory, x){
  "A gives result, B gives memory"
  v = rbind(memory,x)  
  result = A %*% v
  new.memory = B %*% v
  result  = list(result, new.memory)
  return(result)
}

eval.lin.mem.prev = function(A, B, memory.prev, x, x.prev){
  "A gives result, B gives memory"
  v.prev = rbind(memory.prev, x.prev)
  memory = B %*% v.prev
  v = rbind(memory,x)  
  result = A %*% v
  return(result)
}

gradients = function(A,B, mem.current, mem.prev, x.current, x.prev, y){
    # first let's try for grad A
    # Lets do a bunch of asserts about the sizes involved
    # look at output by hand to test
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

# Do line search to find step size
step.size = function(A,B,mem.prev, x.cur, x.prev, y, A.grad, B.grad){
   "See
   https://en.wikipedia.org/wiki/Backtracking_line_search 
    for a reference
   "
   indexes = 1:100
   c = 0.5
   f.A = c(A)
   m = sum(f.A*f.A)
   f.B = c(B)
   m = m + sum(f.B*f.B)
   tee = c*m
   alpha = 0.0020
   tau = 0.5
   response  = eval.lin.mem.prev(A,
                    B, 
                    mem.prev, 
                    x.cur, x.prev)
   error = sum((response - y)^2)
   print("orriginal error")
   print(error)
   print("more errors")
   for(i in indexes){
       new.response  = eval.lin.mem.prev(A+alpha*A.grad,
                        B+alpha*B.grad, 
                        mem.prev, 
                        x.cur, x.prev)
        new.error = sum((new.response - y)^2)
        if(error-new.error >= alpha*tee){
            print("last new.error")
            print(new.error)
            print("error diff")
            print(error-new.error)
            print("alpha tee")
            print(alpha*tee)
            result = alpha
            break
        } else {
            print(new.error)
            alpha = tau*alpha
            if(i == max(indexes)){
                stop("Max iteration reached terminating.")
            }
        }
   }
   return(result)
}

optimize.lin = function(A,B, mem.current, mem.prev, x.current, x.prev, y){
    iter = 1:13
    error.history = c()
    response.history = list()
    A.history = list()
    B.history = list()
    A.grad.history = list()
    B.grad.history = list()
    for( i in iter ){
        grads = gradients(A,B, mem.current, mem.prev, x.current, x.prev, y)
        A.grad = grads[[1]]
        B.grad = grads[[2]]
        alpha = step.size(A,B,mem.prev, x.current, x.prev, y, A.grad, B.grad)
        #alpha = 0.01/i#0.5^(i+5)
        A = A + alpha*A.grad
        B = B + alpha*B.grad
        new.response  = eval.lin.mem.prev(A,
                        B, 
                        mem.prev, 
                        x.current, x.prev)
        new.error = sum((new.response - y)^2)
        error.history = c(error.history, new.error)
        response.history = cbind(response.history, new.response)
        A.history = cbind(A.history, A)
        B.history = cbind(B.history,B)
        A.grad.history = cbind(A.grad.history, A.grad)
        B.grad.history = cbind(B.grad.history,B.grad)
    }
    return(list(A,B,
        error.history,response.history,
        A.history,B.history,
        A.grad.history,B.grad.history))
}
