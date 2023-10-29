import seaborn as sns
import matplotlib.pyplot as plt
import arviz as az


def plot_prior(trace: az.data.inference_data.InferenceData,
               to_plot: str = "prior",
               ylabel: str = 'Likehood') -> None:
    """_summary_

    Args:
        trace (az.data.inference_data.InferenceData): _description_
        to_plot (str, optional): _description_. Defaults to "prior".
        ylabel (str, optional): _description_. Defaults to 'Likehood'.
    """
    sample_prior = trace.to_dict()[to_plot]['x'].flatten()
    sns.kdeplot(sample_prior, label=to_plot)
    plt.ylabel(ylabel)


def plot_prior_predictive(trace: az.data.inference_data.InferenceData, 
                          to_plot: str = "prior") -> None:
    """_summary_

    Args:
        trace (az.data.inference_data.InferenceData): _description_
        to_plot (str, optional): _description_. Defaults to "prior".
    """
    sample_prior_pred = trace.to_dict()[to_plot]['k'].flatten()
    sns.kdeplot(sample_prior_pred, cut=0, label=to_plot + ' ' + 'pred')
    sns.histplot(sample_prior_pred, stat='probability', discrete=True,
                    alpha=0.5, label=to_plot + ' ' 'predictive distribution')
    plt.ylabel("Probability")
    plt.legend()
