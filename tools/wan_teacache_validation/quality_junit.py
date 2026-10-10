# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

"""Export measured quality/performance gates to JUnit without hiding failed gates."""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("comparison", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    metrics = json.loads(args.comparison.read_text())
    run = json.loads((args.comparison.parent / "results.json").read_text())["args"]
    name = f"wan_teacache_{run['split']}_pp{run['pp']}_cfg{run['cfg']}"
    suite = ET.Element("testsuite", name=name, tests="4", errors="0", skipped="0")
    gates = [
        ("mean_ssim", 0.95, True),
        ("min_video_ssim", 0.90, True),
        ("mean_temporal_error_ratio", 0.10, False),
        ("latency_ratio", 0.90, False),
    ]
    failures = 0
    for metric, threshold, lower_bound in gates:
        value = metrics[metric]
        case = ET.SubElement(suite, "testcase", classname=name, name=metric)
        passed = value >= threshold if lower_bound else value <= threshold
        if not passed:
            failures += 1
            relation = ">=" if lower_bound else "<="
            ET.SubElement(case, "failure", message=f"{value:.8g}; required {relation} {threshold}")
    suite.set("failures", str(failures))
    properties = ET.SubElement(suite, "properties")
    ET.SubElement(properties, "property", name="comparison", value=str(args.comparison))
    ET.SubElement(properties, "property", name="videos", value=str(len(metrics["videos"])))
    ET.ElementTree(suite).write(args.output, encoding="utf-8", xml_declaration=True)
    print(json.dumps({"passed": 4 - failures, "failed": failures, "skipped": 0, "report": str(args.output)}))


if __name__ == "__main__":
    main()
