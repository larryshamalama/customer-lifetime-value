import aesara.tensor as at
import numpy as np
import pandas as pd

from aesara.tensor import TensorVariable
from aesara.tensor.random.op import RandomVariable

import pymc as pm
from pymc.distributions.continuous import PositiveContinuous
from pymc.distributions.dist_math import betaln, check_parameters

import matplotlib.pyplot as plt
import arviz as az


class IndividualLevelCLVRV(RandomVariable):
    name = "individual_level"
    ndim_supp = 1
    ndims_params = [0, 0, 0, 0]
    dtype = "floatX"
    _print_name = ("IndividualLevelCLV", "\\operatorname{IndividualLevelCLV}")

    def make_node(self, rng, size, dtype, lam, p, T, T0):

        T = at.as_tensor_variable(T)
        T0 = at.as_tensor_variable(T0)

        # T0 and T cannot be random variables
        if T.owner is not None:
            raise ValueError("T must be a scalar, i.e. observed, and not random quantities.")

        if T0.owner is not None:
            raise ValueError("T0 must be a scalar, i.e. observed, and not random quantities.")

        return super().make_node(rng, size, dtype, lam, p, T, T0)

    def __call__(self, lam, p, T, T0=0, size=None, **kwargs):
        return super().__call__(lam, p, T, T0, size=size, **kwargs)
    
    def _infer_shape(self, size, dist_params, param_shapes=None):
        size = tuple(size)
        
        return size + (2,)

    @classmethod
    def rng_fn(cls, rng, lam, p, T, T0, size) -> np.array:

        if size is None:
            size = ()
        elif isinstance(size, int):
            size = (size,)
        else:
            size = tuple(size)
        
        # To do: broadcast sizes
        lam = np.array(lam)
        p = np.array(p)
        T = np.array(T)
        T0 = np.array(T0)

        param_shape = np.broadcast_shapes(lam.shape, p.shape, T.shape, T0.shape)
        size = param_shape + size

        lam = np.broadcast_to(lam, size)
        p = np.broadcast_to(p, size)
        T = np.broadcast_to(T, size)
        T0 = np.broadcast_to(T0, size)
        
        output = np.empty(shape=size + (2,))
        
        def sim_data(lam, p, T, T0):
            t = 0
            n = 0
            
            while True:
                wait = rng.exponential(scale=1/lam)
                dropout = rng.binomial(n=1, p=p)
                
                if t + wait > T:
                    break
                else:
                    t += wait
                    n += 1
                    
                    if dropout == 1:
                        break
            
            return np.array(
                [
                    t,
                    n,
                ],
            )
        
        for index in np.ndindex(*size):
            output[index] = sim_data(lam[index], p[index], T[index], T0[index])
        
        return output


individual_level_clv = IndividualLevelCLVRV()
            
            
class IndividualLevelCLV(PositiveContinuous):
    rv_op = individual_level_clv
    
    @classmethod
    def dist(cls, lam, p, T, T0, **kwargs):
        return super().dist([lam, p, T, T0], **kwargs)
    
    def get_moment(rv, size, lam, p, T, T0):
        if size is None:
            size = (2,)
        elif isinstance(size, int):
            size = (size,) + (2,)
        else:
            size = tuple(size) + (2,)
        
        return at.full(size, at.as_tensor_variable([lam*(T - T0), 1/r*p]))
    
    def logp(value, lam, p, T, T0):
        t_x = value[..., 0]
        x = value[..., 1]
        
        zero_observations = at.eq(x, 0)
        
        A = x*at.log(1 - p) + x*at.log(lam) - lam*T
        B = at.log(p) + (x - 1)*at.log(1 - p) + x*at.log(lam) - lam*t_x
        
        logp = at.switch(
            zero_observations,
            A,
            at.logaddexp(A, B),
        )
        
        return check_parameters(
            logp,
            lam > 0,
            0 <= p,
            p <= 1,
            at.all(T0 < T),
            msg="lam > 0, 0 <= p <= 1, T0 < T",
        )


class RandomIndividualCLVRV(RandomVariable):
    name = "random_individual_clv"
    ndim_supp = 1
    ndims_params = [0, 0, 0, 0, 0, 0] # a, b, alpha, r, T, T0
    dtype = "floatX"
    _print_name = ("RandomIndividualCLV", "\\operatorname{RandomIndividualCLV}")
    
    def make_node(self, rng, size, dtype, a, b, r, alpha, T, T0):
        T = at.as_tensor_variable(T)
        T0 = at.as_tensor_variable(T0)

        # T0 and T cannot be random variables
        if T.owner is not None:
            raise ValueError("T must be a scalar, i.e. observed, and not random quantities.")

        if T0.owner is not None:
            raise ValueError("T0 must be a scalar, i.e. observed, and not random quantities.")

        return super().make_node(rng, size, dtype, a, b, r, alpha, T, T0)
    
    def __call__(self, a, b, r, alpha, T, T0=0, size=None, **kwargs):
        return super().__call__(a, b, r, alpha, T, T0, size=size, **kwargs)
    
    def _infer_shape(self, size, dist_params, param_shapes=None):
        size = tuple(size)
        
        return size + (2,)
    
    @classmethod
    def rng_fn(cls, rng, a, b, r, alpha, T, T0, size):
        p = rng.beta(a, b, size=size)
        lam = rng.gamma(r, 1/   alpha, size=size)

        return individual_level_clv.rng_fn(rng, lam, p, T, T0, size=None)
    

random_individual_clv = RandomIndividualCLVRV()


class RandomIndividualCLV(PositiveContinuous):
    rv_op = random_individual_clv
    
    @classmethod
    def dist(cls, a, b, r, alpha, T, T0, **kwargs):
        return super().dist([a, b, r, alpha, T, T0], **kwargs)
    
    def get_moment(rv, size, a, b, r, alpha, T, T0):
        return at.full(size, at.as_tensor_variable([3, 3]))
    
    def logp(value, a, b, r, alpha, T, T0):
        t_x = value[..., 0]
        x = value[..., 1]
        
        zero_observations = at.eq(x, 0)
        
        A = betaln(a, b + x) - betaln(a, b) + at.gammaln(r + x) - at.gammaln(r)
        A += r*at.log(alpha) - (r + x)*at.log(alpha + T)
        
        B = betaln(a + 1, b + x - 1) - betaln(a, b) + at.gammaln(r + x) - at.gammaln(r)
        B += r*at.log(alpha) - (r + x)*at.log(alpha + t_x)
        
        logp = at.switch(
            zero_observations,
            A,
            at.logaddexp(A, B),
        )
        
        return check_parameters(
            logp,
            a > 0,
            b > 0,
            alpha > 0,
            r > 0,
            at.all(T0 < T),
            msg="a, b, alpha, r > 0",
        )
