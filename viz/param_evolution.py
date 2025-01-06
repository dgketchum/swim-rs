import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm


def plot_parameter_histograms(initial_params_file, optimized_params_files, fig_out_dir=None):
    """"""
    init = pd.read_csv(initial_params_file, index_col=0)
    init = init[['value']]
    init.index = ['_'.join(i.split('_')[:-1]) for i in init.index]

    sns.set_theme(style="white", context="notebook", font_scale=1.25)
    plt.style.use('dark_background')

    if fig_out_dir:
        for i, param in enumerate(init.index):
            fig, ax = plt.subplots(figsize=(10, 4))
            for j, file in enumerate(optimized_params_files):
                df = pd.read_csv(file, index_col=0)
                cols = ['_'.join(c.split(':')[1].split('_')[1:]) for c in df.columns]
                cols = ['_'.join(c.split('_')[:-2]) for c in cols]
                df.rename(columns={oc: nc for oc, nc in zip(df.columns.tolist(), cols)}, inplace=True)

                sns.kdeplot(df[param], ax=ax, fill=True, label=f'Step {j + 1}')

            ax.axvline(init.loc[param, 'value'], color='red', linestyle='dashed', linewidth=1)
            ax.set_title(param)
            ax.legend()

            plt.tight_layout()

            os.makedirs(fig_out_dir, exist_ok=True)

            fig_path = os.path.join(fig_out_dir, f'{param}.png')
            plt.savefig(fig_path)
            plt.close(fig)
            print(os.path.basename(fig_path))

    else:
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        axes = axes.flatten()

        for i, param in enumerate(['aw', 'tew', 'swe_alpha', 'swe_beta']):
            ax = axes[i]
            for j, file in enumerate(optimized_params_files):
                df = pd.read_csv(file, index_col=0)
                cols = ['_'.join(c.split(':')[1].split('_')[1:]) for c in df.columns]
                cols = ['_'.join(c.split('_')[:-2]) for c in cols]
                df.rename(columns={oc: nc for oc, nc in zip(df.columns.tolist(), cols)}, inplace=True)

                sns.kdeplot(df[param], ax=ax, fill=True, label=f'Step {j + 1}')

            ax.axvline(init.loc[param, 'value'], color='red', linestyle='dashed', linewidth=1)
            ax.set_title(param)
            ax.legend()

        plt.tight_layout()
        plt.show()

if __name__ == '__main__':
    home = os.path.expanduser('~')
    root = os.path.join(home, 'PycharmProjects', 'swim-rs')

    project_ws = os.path.join(root, 'tutorials', '3_Crane')

    data = os.path.join(project_ws, 'data')
    pst = os.path.join(project_ws, 'pest')

    inital_params = os.path.join(project_ws, 'params.csv')
    steps = [os.path.join(pst, f'3_Crane.{i}.par.csv') for i in range(4)]
    fig_dir = os.path.join(project_ws, 'figures', 'parameter_hist')

    if not os.path.isdir(fig_dir):
        os.makedirs(fig_dir, exist_ok=True)

    plot_parameter_histograms(inital_params, steps, fig_dir)
# ========================= EOF ====================================================================
