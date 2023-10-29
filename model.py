import pymc as pm
import arviz as az


def coin_model(a: int,
               b: int,
               n: int,
               seed: int,
               to_sample: str = "prior",
               samples: int = 1000,
               observed: int  | list | None = None
               ) -> tuple[pm.model.core.Model,
                          az.data.inference_data.InferenceData]:
    """_summary_

    Args:
        a (int): _description_
        b (int): _description_
        n (int): _description_
        seed (int): _description_
        to_sample (str, optional): _description_. Defaults to "prior".
        samples (int, optional): _description_. Defaults to 1000.
        observed (int | list | None, optional): _description_. Defaults to None.

    Returns:
        tuple[pm.model.core.Model, arviz.data.inference_data.InferenceData]: _description_
    """
    with pm.Model() as model:
        x = pm.Beta('x', alpha = a, beta = b)
        k = pm.Binomial('k', n = n, p = x, observed=observed)
        if to_sample == "prior":
            trace = pm.sample_prior_predictive(samples, random_seed=seed)
        elif to_sample == "posterior":
            trace = pm.sample(samples, random_seed=seed)
        else:
            trace = None
    return model, trace
