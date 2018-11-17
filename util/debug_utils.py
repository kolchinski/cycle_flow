import torch

# Given a torch tensor t, follow its gradients
# all the way up the graph
# Returns a list of all the ancestor backward nodes
def ancestor_grads(t):
    l = []
    fn = t.grad_fn
    while fn is not None:
        l.append(fn)
        fn = fn.next_functions[0][0]
    return l