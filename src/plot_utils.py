"""Plot utilities"""

def get_plot_configs() -> dict:
    """Get the configuration for pyplot

    :returns: rcParams compatible config (dict)

    """
    return {
        "axes.formatter.use_mathtext": True,
        "font.size": 14,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "font.family": "serif",
        # "font.sans-serif": ["Computer Modern Roman"],
        # "text.usetex": True,
    }
