#
#
#
# Copyright (C) 2020 Zach (Yuzhe) Ni [First name] [Last Name]
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2, or (at your option)
# any later version.
#
#
# This python file will introduce the concept of tensor, which is a useful tool in dealing with different physics transformation
#
import numpy as np
import itertools


# helper method for constructing permutation Matrix
def constructPermutationMatrix(permutation):
  dimensionN = permutation.shape[0]
  perm = np.zeros((dimensionN,dimensionN))
  for i in range(0,dimensionN):
    perm[i][permutation[i]] = 1
  return perm

class Tensor:
    # We construct our tensor to be mapping from V^k to R^n where k = domainDimension and n = rangeDimension
    # The argument function takes a vector v in V^k as an input and it will product a vector u in R^n
    def __init__(self, domainDimension, rangeDimension, function):
        self.k = domainDimension
        self.n = rangeDimension
        self.f = function
    # we want to check if f(ax+y) = af(x)+f(y) where a and b are scalars
    # we can use monte carlo simulation here
    # we will also be using matrix solver to find exact linear relation between v and u and then verify or 
    # not implemented yet
    def checkLinearity(self):
        return True
        pass
    # we will check if a tensor is alternating or not by inspecting if we swap two components of our function v to see how it will change the sign of u. 
    # We will still use monte Carlo method here
    # The checking of sign change is cheap, we just add the two vectors up to see if they induce a zero vector. 
    def checkIfAlternating(self):
        return None
        pass
    # format of permuation [0,1,2] means identity permutation
    # arg self: a tensor
    # arg permuation: a numpy array of length self.k and it contains all the number for 0 to self.k-1
    # We can easily see that by permuting we will still get a tensor of dimension k x n
    def inducePermuationTensor(self, permutation):
        def perm(x):
          y = np.zeros(x.shape[0])
          for i in range(0,x.shape[0]):
            y[permutation[i]] = x[i]
          return self.f(y)
        tor = Tensor(self.k,self.n,perm)
        return tor
    # scalar product
    # arg scalar: a scalar
    def __mul__(self, scalar):
       return Tensor(self.k,self.n,lambda x : self.f(x)*scalar)
    # Tensor Addition
    def __add__(self, tensorG):
       return Tensor(self.k,self.n, lambda x : self.f(x) + tensorG.f(x))
    # Compute alternating tensor of our f
    def computeAlt(self):
      list_x = list(itertools.permutations(np.asarray([i for i in range(0,self.k)])))
      ret = Tensor(self.k, self.n, lambda x: np.zeros(self.n))
      for x in list_x:
         ax = np.asarray(x)
         evenorodd = np.linalg.det(constructPermutationMatrix(ax))
         ret = ret + (self.inducePermuationTensor(ax)*evenorodd)
         #print("permutation: " + str(ax) + " EvenorOdd: " + str(evenorodd)  + "Value: " + str(self.inducePermuationTensor(ax).f(np.asarray([1,2,3]))))
      num = len(list_x)
      num = 1.0/ num
      return ret*num
    # we construct our tensor product
    # remember the definition of a tensorproduct require us to have f and g are both tensors over V and we will have the range dimension to be the same for f and g
    def Product(self,tensorG):
        if not self.checkLinearity():
          raise Exception('The tensor f is not a valid tensor by violating linearity.')
        if not tensorG.checkLinearity():
          raise Exception('The tensor g is not a valid tensor by violating linearity.')
        if self.n != tensorG.n:
          raise Exception('The two tensors don\'t agree on output dimension.')
        # otherwise, we will return a tensor  with k = f.k + g.k and rangeDimension = 1
        # function (x) np.dot(f.function(x[0:f.k]),g.function(x[f.k:f.k+g.k]))
        TensorProd = lambda x : np.dot(self.f(x[0:self.k]),tensorG.f(x[self.k:self.k+tensorG.k]))
        
        return Tensor(self.k+tensorG.k, 1, TensorProd)
