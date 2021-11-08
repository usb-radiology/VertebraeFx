import argparse
import json
from datetime import datetime
from pathlib import Path

from environs import Env
from jinja2 import Environment, FileSystemLoader, select_autoescape

from report.generator import report

env = Env()
env.read_env()  # read .env file, if it exists
templates_path = env("TEMPLATES_PATH", "/data/software/VertebraeFx/report")

env = Environment(
    loader=FileSystemLoader(templates_path),
    autoescape=select_autoescape(["html", "xml"]),
)


def to_date(date_as_int):
    if date_as_int:
        return datetime.strptime(str(date_as_int), "%Y%m%d").strftime("%d.%m.%Y")
    else:
        return ""


def now():
    return datetime.now().strftime("%d.%m.%Y %H:%M:%S")


env.filters["to_date"] = to_date
env.globals["now"] = now
version = "0.4"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", dest="model", required=True, help="Model path")
    parser.add_argument("-i", dest="image", required=True, help="Image path")
    parser.add_argument(
        "-s", dest="segmentation", required=True, help="Segmentation path"
    )
    parser.add_argument("-o", dest="output_dir", required=True, help="Output folder")
    parser.add_argument("--version", action="version", version=f"{version}")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) / "predictions"
    output_dir.mkdir(exist_ok=True)
    print(f"Saving outputs to dir: {output_dir}")

    if (Path(args.image).parent.parent / "nodeinfo.json").exists():
        with open(Path(args.image).parent.parent / "nodeinfo.json", "r") as f:
            metadata = json.load(f)
    else:
        metadata = {
            "PatientName": "Unknown",
            "PatientSex": "Unknown",
            "PatientBirthDate": "19000101",
            "PatientID": "Unknown",
            "StudyID": "Unknown",
            "StudyDescription": "Unknown",
            "StudyDate": "19000101",
        }
    report(args, env, metadata, version)


if __name__ == "__main__":
    main()
