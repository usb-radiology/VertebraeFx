from pathlib import Path
from timeit import default_timer as timer

import imgkit
import nibabel as nib
import numpy as np
from common.vertebrae import Vertebrae

from report.image_slicer import generate_images, load_model, predict
from report.overview import create_overview_image


def load(image, mask):
    pred_cor_data = np.squeeze(np.rot90(np.flipud(nib.load(mask).get_fdata()), 3))
    raw_data = np.rot90(np.flipud(nib.load(image).get_fdata()), 3)
    return raw_data, pred_cor_data


def process(image_path, segmentation_path, output):
    image, segmentation = load(image_path, segmentation_path)
    mask_values = [int(x) for x in np.unique(segmentation.ravel()) if x > 0]
    all_vertebraes = sorted(list({Vertebrae(i) for i in mask_values}))
    overview_image = create_overview_image(image, segmentation, all_vertebraes, output)
    v_mapping = {}
    for v in all_vertebraes:
        v_mapping[v] = generate_images(image, segmentation, v)

    return v_mapping, overview_image


def report(args, env, metadata, version):
    start = timer()
    output_dir = (Path(args.output_dir) / "predictions").resolve()
    v_mappings, overview_image = process(args.image, args.segmentation, output_dir)
    m = load_model(args.model)
    predictions = predict(m, v_mappings)

    start = timer()
    t = env.get_template("template.html")
    output_html = t.render(
        version=version,
        predictions=predictions,
        overview_image=overview_image,
        metadata=metadata,
    )
    end = timer()
    print(f"HTML template creation took: {(end-start):.3f} sec")
    options = {"xvfb": "", "format": "png", "width": "1200", "height": "1200"}
    print("Start generating screenshot ...")
    report_file = output_dir / "vertebrae_fracture_report.png"
    imgkit.from_string(output_html, str(report_file), options=options)
    print(f"Screenshot saved to {report_file}")
    return 0
