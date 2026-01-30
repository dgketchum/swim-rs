import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from swimrs.process.state import TUNABLE_PARAMS


def plot_parameter_histograms(initial_params_file, optimized_params_files, fig_out_dir=None):
    """"""
    df = None

    init = pd.read_csv(initial_params_file, index_col=0)

    sites = [[i.replace(f"{p}_", "") for i in init.index if p in i] for p in TUNABLE_PARAMS[:1]][0]

    for site in sites:
        if site != "ALARC2_Smith6":
            continue

        sns.set_theme(style="white", context="notebook", font_scale=1.25)
        plt.style.use("dark_background")

        if fig_out_dir:
            for i, param in enumerate(TUNABLE_PARAMS):
                fig, ax = plt.subplots(figsize=(10, 4))
                for j, file in enumerate(optimized_params_files):
                    df = pd.read_csv(file, index_col=0)
                    col = [c for c in df.columns if param in c and site.lower() in c]
                    assert len(col) == 1
                    sns.kdeplot(df[col], ax=ax, fill=True, label=f"Step {j + 1}")

                ax.axvline(df[col].values.mean(), color="red", linestyle="dashed", linewidth=1)
                ax.set_title(param)
                ax.legend()

                plt.tight_layout()

                os.makedirs(fig_out_dir, exist_ok=True)

                fig_path = os.path.join(fig_out_dir, f"{site}_{param}.png")
                plt.savefig(fig_path)
                plt.close(fig)
                print(os.path.basename(fig_path))

        else:
            fig, axes = plt.subplots(2, 2, figsize=(12, 12))
            axes = axes.flatten()

            for i, param in enumerate(["aw", "tew", "swe_alpha", "swe_beta"]):
                ax = axes[i]
                for j, file in enumerate(optimized_params_files):
                    df = pd.read_csv(file, index_col=0)
                    cols = ["_".join(c.split(":")[1].split("_")[1:]) for c in df.columns]
                    cols = ["_".join(c.split("_")[:-2]) for c in cols]
                    df.rename(
                        columns={oc: nc for oc, nc in zip(df.columns.tolist(), cols)}, inplace=True
                    )

                    sns.kdeplot(df[param], ax=ax, fill=True, label=f"Step {j + 1}")

                ax.axvline(init.loc[param, "value"], color="red", linestyle="dashed", linewidth=1)
                ax.set_title(param)
                ax.legend()

            plt.tight_layout()
            plt.show()


if __name__ == "__main__":
    home = os.path.expanduser("~")
    root = os.path.join(home, "code", "swim-rs")

    project = "alarc_test"
    # project = '3_Crane'

    project_ws = os.path.join(root, "examples", project)

    data = os.path.join(project_ws, "data")
    master = os.path.join(project_ws, "master")

    inital_params = os.path.join(project_ws, "params.csv")
    steps = [os.path.join(master, f"{project}.{i}.par.csv") for i in range(3)]
    fig_dir = os.path.join(project_ws, "figures", "parameter_hist")

    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    plot_parameter_histograms(inital_params, steps, fig_dir)
# ========================= EOF ====================================================================
