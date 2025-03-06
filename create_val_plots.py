# Standard library
import pickle
from pathlib import Path

# Third-party
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_DIR = "val_plots/graph_design"

# Configuration
METRICS_FILES = {
    # NO FINETUNING
    # "7.19° - no finetune": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250304_111043-qvzebmz8/files/test_metrics.pkl",
    # DEFAULT + BOUNDARY WIDTH
    # "no boundary": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250223_135811-taoe2q20/files/test_metrics.pkl",
    # "3.6°": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250223_135710-5ikae3ta/files/test_metrics.pkl",
    "7.19°": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250223_135748-rx3r2qc1/files/test_metrics.pkl",
    # "7.19° -- time embed": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250228_193408-rbp9ta35/files/test_metrics.pkl",
    # "7.19° - no future": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250223_203137-pevaufbw/files/test_metrics.pkl",
    # "7.19° with interior": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250223_094539-ve7jxmni/files/test_metrics.pkl",
    # "10.79°": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250223_194442-1mrvluca/files/test_metrics.pkl",
    # "14.39°": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250223_135712-d82t2arn/files/test_metrics.pkl",
    # GRAPH DESIGN
    "Triangular": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250305_143223-sc0ind3i/files/test_metrics.pkl",
    # SUBSAMPLING
    # "7.19° with 3h subsample": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250223_135759-liu0vlsa/files/test_metrics.pkl",
    # FINETUNING
    # "LR 1e-4": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250223_135732-szuf2x6a/files/test_metrics.pkl",
    # "LR 1e-4 - AR 12": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250223_135723-klfmyn8q/files/test_metrics.pkl",
    # "LR 1e-4 - AR 6": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250223_135727-bkrji8ip/files/test_metrics.pkl",
    # IFS BOUNDARY (BUGGED)
    # "7.19 IFS margin with interior": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250223_122208-9h4qffmp/files/test_metrics.pkl",
    # "7.19 IFS margin with interior -- LR 0.0001": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250223_122212-mllsk83b/files/test_metrics.pkl",
    # "7.19 IFS margin with interior -- LR 0.0001 -- AR 12": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250223_122219-0c4wc8gs/files/test_metrics.pkl",
    # "7.19 IFS margin with interior -- LR 0.0001 -- AR 6": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250223_122215-4euaxytl/files/test_metrics.pkl",
    # "7.19 IFS margin": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250223_122204-2po0e9pl/files/test_metrics.pkl",
    # "7.19 IFS margin -- dynamic time embedding": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250228_193433-dokp0m5e/files/test_metrics.pkl",
    # IFS BOUNDARY BUGFIX
    # "7.19 IFS": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250227_173313-wvauu4xx/files/test_metrics.pkl",
    # "7.19 with interior IFS": "/iopsstor/scratch/cscs/sadamov/pyprojects_data/neural-lam/wandb/run-20250227_173315-ogt0lmto/files/test_metrics.pkl",
}

VARIABLES = {
    "T_2M": "2m_temperature",
    "U_10M": "10m_u_component_of_wind",
    "V_10M": "10m_v_component_of_wind",
    "PMSL": "mean_sea_level_pressure",
    "PS": "surface_pressure",
    "TOT_PREC": "total_precipitation",
    "ASOB_S": "surface_net_shortwave_radiation",
    "ATHB_S": "surface_net_longwave_radiation",
}

WIND_PAIRS = {
    "wv10m": ("U_10M", "V_10M"),
}

# Colorblind-friendly palette (based on Wong's Nature Methods 2011 & Okabe-Ito)
COLORS = {
    "blue": "#56B4E9",  # Sky blue
    "orange": "#E69F00",  # Orange
    "green": "#009E73",  # Bluish green
    "red": "#D55E00",  # Vermilion
    "purple": "#CC79A7",  # Reddish purple
    "yellow": "#F0E442",  # Yellow
    "grey": "#999999",  # Grey
    "black": "#000000",  # Black
    "cyan": "#0072B2",  # Blue
    "brown": "#8C510A",  # Brown
}

# Line styles that are distinguishable
LINE_STYLES = [
    "solid",  # ___________
    "dashed",  # - - - - - -
    "dashdot",  # -.-.-.-.-.-
    "dotted",  # .............
    (0, (3, 1)),  # ...  ...  ...
    (0, (5, 1)),  # ....    ....
    (0, (1, 1)),  # . . . . . .
    (0, (3, 1, 1, 1)),  # -..-..-..
    (0, (3, 1, 1, 1, 1, 1)),  # -..-..-.
    (0, (1, 2, 5, 2)),  # Complex dash
]

# Distinct markers
MARKERS = [
    "o",  # Circle
    "s",  # Square
    "D",  # Diamond
    "^",  # Triangle up
    "v",  # Triangle down
    "P",  # Plus (filled)
    "X",  # X (filled)
    "p",  # Pentagon
    "*",  # Star
    "h",  # Hexagon
]

UNIT_LOOKUP = {
    "m s**-1": "m / s",
    "W m**-2": "W / m^2",
    "m**-2": "W / m^2",
    "m**2 s**-2": "m^2 / s^2",
}

VARIABLE_UNITS = {
    "U_10M": "m / s",
    "V_10M": "m / s",
    "wv10m": "m / s",
    "T_2M": "K",
    "PMSL": "Pa",
    "PS": "Pa",
    "TOT_PREC": "mm / h",
    "ASOB_S": "W / m^2",
    "ATHB_S": "W / m^2",
}


def create_style_dict(metrics_dict):
    """Create style dictionary for experiments using distinct visual elements"""
    styles = {}
    for i, model_name in enumerate(metrics_dict.keys()):
        styles[model_name] = {
            "color": list(COLORS.values())[i % len(COLORS)],
            "linestyle": LINE_STYLES[i % len(LINE_STYLES)],
            "marker": MARKERS[i % len(MARKERS)],
        }
    return styles


# Update plot_kwargs in plot_metrics function
def get_plot_kwargs(style, model_name, time_step):
    """Get consistent plot kwargs for all plots"""
    return {
        "label": model_name,
        "color": style["color"],
        "linestyle": style["linestyle"],
        "marker": style["marker"],
        "markersize": 4,
        "markevery": int(12 / time_step),  # Every 12 h
        "markerfacecolor": "white",
        "markeredgewidth": 1.0,
        "linewidth": 1.5,
    }


def load_metrics(file_path):
    """Load metrics from pickle file."""
    with open(file_path, "rb") as f:
        return pickle.load(f)


def save_plot(fig, name, time=None, output_dir=None):
    """Save plots to consistent location."""
    if time is not None:
        name = f"{name}_{time.dt.strftime('%Y%m%d_%H').values}"
    if output_dir is None:
        output_dir = "plots"

    fig.savefig(Path(output_dir) / f"{name}.pdf", bbox_inches="tight", dpi=300)
    plt.close()


def plot_metrics(
    metrics_files,
    metric_name="rmse",
    variables=None,
    combined=False,
    output_dir=None,
    wind_pair_vars=None,
    max_lead_time=None,
    legend_placement="best",
):
    """
    Unified plotting function with consistent styling

    wind_pair_vars is map from a variable name to two wind variables to
    derive it from, e.g. {"wv10m": ("u10m", "v10m")}.

    max_lead_time should be None (keep all lead times) or a np.timedelta64
    """
    plt.style.use("default")

    if wind_pair_vars is None:
        wind_pair_vars = {}
    else:
        # Make sure all of wind pair variables are also in variables list
        for wp_var in wind_pair_vars.keys():
            if wp_var not in variables:
                variables.append(wp_var)

    metrics_dict = {
        model_name: load_metrics(file_path).sel(
            lead_time=slice(None, max_lead_time)
        )
        for model_name, file_path in metrics_files.items()
    }

    if output_dir is not None:
        Path(output_dir).mkdir(exist_ok=True, parents=True)

    # Create style dictionary based on number of experiments
    PLOT_STYLES = create_style_dict(metrics_dict)

    if combined:
        n_cols = 2
        n_rows = (len(variables) + n_cols - 1) // n_cols
        _ = plt.figure(figsize=(15, 4 * n_rows))

    for idx, var in enumerate(variables, 1):
        if combined:
            ax = plt.subplot(n_rows, n_cols, idx)
        else:
            _, ax = plt.subplots(figsize=(5, 3), dpi=100)

        for model_name, metrics in metrics_dict.items():
            lead_time_hrs = metrics.lead_time.dt.total_seconds() / 3600
            # Time step in h
            time_step = (
                metrics.lead_time.diff("lead_time")[0]
                .values.astype("timedelta64[h]")
                .astype(int)
            )

            if var in wind_pair_vars:
                # Derive this metric value from pair of wind fields
                field1, field2 = wind_pair_vars[var]
                metric_values = np.sqrt(
                    metrics[metric_name].sel(variable=field1) ** 2
                    + metrics[metric_name].sel(variable=field2) ** 2
                )

                # Same unit as one of the fields above
                var_unit = str(
                    metrics["variable_units"].sel(variable=field1).values
                )
            else:
                metric_values = metrics[metric_name].sel(variable=var)
                var_unit = str(
                    metrics["variable_units"].sel(variable=var).values
                )

            if var_unit in UNIT_LOOKUP:
                var_unit = UNIT_LOOKUP[var_unit]
            if var_unit == "":
                var_unit = VARIABLE_UNITS[var]

            style = PLOT_STYLES[model_name]
            plot_kwargs = get_plot_kwargs(style, model_name, time_step)

            ax.plot(
                lead_time_hrs,
                metric_values,
                **plot_kwargs,
            )

        # Common styling
        ax.set_xlabel("Lead Time (hours)", fontsize=10 if combined else 12)
        ax.set_ylabel(
            f"{metric_name.upper()} (${var_unit}$)",
            fontsize=10 if combined else 12,
        )
        ax.grid(True, linestyle="--", alpha=0.3)
        ax.tick_params(
            axis="both", which="major", labelsize=9 if combined else 10
        )

        if idx == 1:
            ax.legend(
                frameon=True,
                facecolor="white",
                edgecolor="black",
                fontsize=10,
                loc=legend_placement,
            )

        if not combined:
            plt.tight_layout()
            save_plot(plt, f"{var}_{metric_name}", output_dir=output_dir)

    if combined:
        plt.tight_layout()
        save_plot(plt, f"combined_{metric_name}", output_dir=output_dir)


def main():
    variables = list(VARIABLES.keys())
    print("Plotting metrics...")
    plot_metrics(
        METRICS_FILES,
        metric_name="rmse",
        variables=variables,
        output_dir=OUTPUT_DIR,
        combined=False,
        wind_pair_vars=WIND_PAIRS,
        max_lead_time=np.timedelta64(48, "h"),  # Limit to 2 days
    )


if __name__ == "__main__":
    main()
