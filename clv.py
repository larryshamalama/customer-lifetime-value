import numpy as np

from aesara.tensor.random.op import RandomVariable

import pymc as pm

from pymc.distributions.distributions import Discrete
from pymc.distributions.continuous import PositiveContinuous


class RightCensoredNegativeBinomialRV(RandomVariable):
    name = "RightCensoredNegativeBinomial"
    ndim_supp = 0
    ndims_params = [0, 0, 0, 0]
    dtype = "floatX"
    _print_name = ("RightCensoredNegBinom", "\\{RightCensoredNegBinom}")

    @classmethod
    def rng_fn(cls, rng, size, lam, p, T, T0):
        r = rng.geometric(p=p, size=size)
        n = rng.poisson(lam*(T - T0), size=size)

        x = np.minimum(r, n) # elements are min(r, n)

        return x


right_censored_negative_binomial = RightCensoredNegativeBinomialRV()


class RightCensoredNegativeBinomial(Discrete):
    rv_op = right_censored_negative_binomial()
    
    @classmethod
    def dist(cls, lam, p, T, T0, *args, **kwargs):
        return

    def get_moment(rv, size, lam, p, T, T0):
        pass

    def logp(value, lam, p, T, T0):
        active_after_x = logpow(lam * T * (1 - p), value) - lam * T - factln(value)
        inactive_or_censored = at.log(p) + logpow(1 - p, value - 1)
        inactive_or_censored += at.log1p(- at.cumsum())


class UniformOrderStatisticRV(RandomVariable):
    name = "UniformOrderStatistic"
    ndim_supp = 0
    ndims_params = [0, 0, 0, 0]
    dtype = "floatX"
    _print_name = ("")

    @classmethod
    def rng_fn(cls, rng, size, k, n, lower, upper):
        # needs broadcasting
        return rng.beta(k, n - k + 1, size=size)*(upper - lower) + lower


uniform_order_statistic = UniformOrderStatisticRV()


class UniformOrderStatistic(PositiveContinuous):
    rv_op = uniform_order_statistic

    @classmethod
    def dist(cls, k, n, lower, upper, *args, **kwargs):
        return super().dist([k, n, lower, upper], **kwargs)

    def get_moment(rv, size, k, n, lower, upper):
        pass

    def logp(value, k, n, lower, upper):
        pass


class CustomerLifetimeValue:

    def __init__(self, tx, frequency, r, alpha, a, b, T, T0):
        lam = pm.Gamma(name="lam", alpha=r, beta=alpha)
        p = pm.Beta(name="p", alpha=a, beta=b)

        frequency = RightCensoredNegativeBinomial(name="frequency", lam=lam, p=p, T=T, T0=0, observed=frequency)
        tx = UniformOrderStatistic(name="tx", k=n, n=n, lower=T0, upper=T, observed=tx)


if __name__ == "__main__":

    from lifetimes.utils import summary_data_from_transaction_data
    from lifetimes.datasets import load_dataset

    rfm = summary_data_from_transaction_data(
        cdnow_transactions,
        "customer_id",
        "date",
        observation_period_end=pd.to_datetime("1997-09-30"),
        freq='W'
    )

    frequency = rfm["frequency"].to_numpy()
    tx = rfm["recency"].to_numpy()
    T = rfm["T"].to_numpy()

    with pm.Model() as model:
        # hyper priors for the Beta prior
        r = pm.HalfNormal(name="r", sigma=10)
        alpha = pm.HalfNormal(name="alpha", sigma=10)

        # hyper priors for the Gamma prior
        a = pm.HalfNormal(name="a", sigma=10)
        b = pm.HalfNormal(name="b", sigma=10)

        clv = CustomerLifetimeValue(tx=tx, frequency=frequency, r=r, alpha=alpha, a=a, b=b, T=T, T0=0)

        prior = pm.sample_prior_predictive()
        idata = pm.sample(samples=10000, chains=1)
