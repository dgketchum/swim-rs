import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def split_path(path):
    parts = []
    while True:
        path, folder = os.path.split(path)
        if folder:
            parts.append(folder)
        else:
            if path:
                parts.append(path)
            break
    return parts[::-1]


def run():
    """This script is meant to be executed by PEST++"""

    path = split_path(__file__)
    project_top = path.index("swim-rs")
    root = os.path.join(*path[: project_top + 1])

    os.environ["PYTHONPATH"] = root

    model_script = os.path.join(root, "run", "run_mp.py")

    project_ws = os.path.join(root, "examples", "DUMMY_PROJECT")

    conf_file = os.path.join(project_ws, "config.toml")

    cwd = os.getcwd()

    calibration_dir = os.path.join(cwd, "mult")

    args = [
        "python" + f" {model_script}",
        "--project_dir",
        project_ws,
        "--config_path",
        conf_file,
        "--worker_dir",
        cwd,
        "--calibration_dir",
        calibration_dir,
    ]

    os.system(" ".join(args))


if __name__ == "__main__":
    run()
