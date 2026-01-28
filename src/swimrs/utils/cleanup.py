import json
import os


def cleanup(clean_dir, in_json):
    with open(in_json) as fp:
        missing = json.load(fp)

    for sid in missing["irr_partial"]:
        f = os.path.join(clean_dir, f"{sid}_output.csv")
        if os.path.exists(f):
            os.remove(f)
            print(f"removed {f}")
        else:
            print(f"{f} doesnt exist")

    a = 1


if __name__ == "__main__":
    root = "/media/research/IrrigationGIS/swim"
    if not os.path.exists(root):
        root = "/home/dgketchum/data/IrrigationGIS/swim"

    project = "tongue"

    data = os.path.join(root, "examples", project, "output")
    output = os.path.join(data, "irr_output")
    js = os.path.join(data, "missing.json")
    cleanup(output, js)

# ========================= EOF ====================================================================
