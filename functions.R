check.optimize.input = function(A,B,xs,ys){
    #browser()
    if(class(xs) != "list"){
        stop("xs is not a list")
    }
    if(class(ys) != "list"){
        stop("ys is not a list")
    }
    if(length(xs) != length(ys)){
        stop("Lengths of xs and ys don't match")
    } 
    if (length(xs) <=1) {
        stop("Length of xs needs to be two or more")
    } 
    if(class(A) != "matrix"){
        stop("A is not a matrix")
    }
    if(class(B) != "matrix"){
        stop("B is not a matrix")
    }
    if (any(sapply(ys, class) != "matrix")){
        stop("some y isn't a matrix")
    } 
    if (any(sapply(xs, class) != "matrix")){
        stop("some x isn't a matrix")
    } 
    if (any(sapply(ys, ncol) != 1)){
        stop("some y  doesn't have exactly one column")
    } 
    if (any(sapply(xs, ncol) != 1)){
        stop("some x  doesn't have exactly one column")
    } 
    if (any(sapply(ys, nrow) != nrow(ys[[1]]))){
        stop("some y doesn't have same number of rows ")
    } 
    if (any(sapply(xs, nrow) != nrow(xs[[1]]))){
        stop("some x doesn't have same number of rows ")
    }
    nr.x  = nrow(xs[[1]])
    nr.y = nrow(ys[[1]])
    nr.m = nrow(B)
    if(nrow(A) != nr.y) {
        stop("output size of A doesn't match size of y")
    } 
    if(nr.m + nr.x != ncol(A)) {
        stop("input size of A dosent match size of x plus m")
    } 
    if(nr.m + nr.x != ncol(B)) {
        stop("input size of B doesn't match size of x plus m")
    }
    print("done with checks")
}

mk.v = function(mem, x){
    return(rbind(mem,x))
}

eval.lin.mem = function(A, B, mem, x){
  "A gives result, B gives memory"
  v = mk.v(mem,x)  
  result = A %*% v
  new.mem = B %*% v
  result  = list(result, new.mem)
  return(result)
}

eval.lin.mem.prev = function(A, B, mem.prev, x, x.prev){
  "A gives result, B gives memory"
  v.prev = mk.v(mem.prev, x.prev)
  mem = B %*% v.prev
  v = mk.v(mem, x)  
  result = A %*% v
  return(result)
}

gradients = function(A,B, mem.cur, mem.prev, x.cur, x.prev, y){
    # first let's try for grad A
    # Lets do a bunch of asserts about the sizes involved
    # look at output by hand to test
    results = eval.lin.mem(A,B, mem.cur, x.cur)
    response = results[[1]]
    dy = y - response
    # what goes into the result
    v.response = rbind(mem.cur, x.cur)
    A.grad = dy %*% t(v.response)
    #now we try for grad B
    dm = (t(A) %*% dy)[1:nrow(mem.cur),] #top m rows
    v.mem = rbind(mem.prev, x.prev)
    B.grad = dm %*% t(v.mem)
    result = list(A.grad, B.grad)
    return(result)
}

# Do line search to find step size
step.size = function(A,B,mem.prev, x.cur, 
                    x.prev, y, A.grad, B.grad,
                    tol, max.iter = 100){
   "See
   https://en.wikipedia.org/wiki/Backtracking_line_search 
    for a reference
   "
   c = 0.50
   alpha = 0.5
   tau = 0.5
   f.A = c(A.grad)
   m.A = sum(f.A*f.A)
   f.B = c(B.grad)
   m.B = sum(f.B*f.B)
   m = m.A + m.B
   tee = c*m
   response  = eval.lin.mem.prev(A,
                    B, 
                    mem.prev, 
                    x.cur, x.prev)
   error = sum((response - y)^2)
   for(i in 1:max.iter){
       new.response  = eval.lin.mem.prev(
                            A + alpha*A.grad,
                            B + alpha*B.grad, 
                            mem.prev, 
                            x.cur, x.prev)
        new.error = sum((new.response - y)^2)
        if(new.error <= tol){
            result = alpha
            break
        } else if(error-new.error >= alpha*tee){
            result = alpha
            break
        } else {
            if(i == max.iter){
                if(error > new.error){
                    warning("Unable to satisfy Armijo-Goldstein condition,\
                            but new.error < error, \
                            so we are returning nonetheless.")
                    result = alpha
                } else{
                    stop("Max iteration  reached\
                        for determining step size\
                        and error not improving.")
                }
            } else{
                alpha = tau*alpha
            }
        }
   }
   return(result)
}

optimize.lin = function(A,B, xs, ys, tol=0.001){
    # error might increase because you let it roll once
    check.optimize.input(A,B,xs,ys)
    mem.current = matrix(0, nrow = nrow(B))
    error.history = list()
    response.history = list()
    A.history = list()
    B.history = list()
    A.grad.history = list()
    B.grad.history = list()
    mem.history = list()
    for(i in 2:length(xs)){
        # generate variables
        x.current = xs[[i]]
        x.prev = xs[[i-1]]
        y = ys[[i]]
        mem.prev = mem.current
        mem.current = B %*% mk.v(mem.prev, x.prev)
        # do gradients
        grads = gradients(A,B, mem.current, mem.prev, x.current, x.prev, y)
        A.grad = grads[[1]]
        B.grad = grads[[2]]
        # find step size
        alpha = step.size(A,B,mem.prev, x.current, 
                            x.prev, y, A.grad, B.grad, tol)
        # update matrices
        A = A + alpha*A.grad
        B = B + alpha*B.grad
        mem.current = B %*% mk.v(mem.prev, x.prev)
        # calculate histories
        new.response  = eval.lin.mem.prev(A,
                            B, 
                            mem.prev, 
                            x.current, x.prev)
        new.error = sum((new.response - y)^2)
        error.history = list.append(error.history, new.error)
        response.history = list.append(response.history, new.response)
        A.history = list.append(A.history, A)
        B.history = list.append(B.history, B)
        A.grad.history = list.append(A.grad.history, A.grad)
        B.grad.history = list.append(B.grad.history, B.grad)
        mem.history = list.append(mem.history, mem.current)
        if(new.error <= tol){
            # don't actually want #####
            # want in line search
            print("tolerance reached")
            break
        }
    }
    return(list(A,B,
        error.history,response.history,
        A.history,B.history,
        A.grad.history,B.grad.history,
        "mem" = mem.history))
}

list.append = function(old.list, new.item){
    if(class(old.list) != "list"){
        stop("old.list should be a list in list.append")
    }
    result = unlist(list(old.list,list(new.item)), recursive = FALSE)
    return(result)
}

test1 = function(){
    A = matrix(runif(2), 1, 2)
    B = matrix(runif(2), 1, 2)
    xs = as.list(replicate(10, matrix(1), simplify=FALSE))
    ys = as.list(replicate(10, matrix(5), simplify=FALSE))
    results = optimize.lin(A,B,xs,ys)
    return(results)
}
