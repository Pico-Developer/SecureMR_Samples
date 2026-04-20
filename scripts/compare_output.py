#!/usr/bin/env python3
import argparse
import glob
import os
import struct
import sys


def load_expected(src: str):
    if os.path.isfile(src):
        data = open(src, "rb").read()
        if len(data) % 4 != 0:
            print(f"Expected file {src} size {len(data)} is not divisible by 4 bytes (float32).", flush=True)
        count = len(data) // 4
        return list(struct.unpack("<%df" % count, data[: count * 4]))
    return [float(x) for x in src.split(",") if x.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Compare model_inspect outputs against expected float32 values or file."
    )
    parser.add_argument("expected", help="Comma-separated float32 values or float32 binary file")
    parser.add_argument("output_dir", help="Directory containing model_inspect_output_*.bin")
    parser.add_argument("output_name", nargs="?", default="", help="Specific output file name to compare")
    args = parser.parse_args()

    expected = load_expected(args.expected)
    if not expected:
        print("No expected values to compare; skipping.", flush=True)
        return 0

    files = sorted(glob.glob(os.path.join(args.output_dir, "model_inspect_output_*.bin")))
    if not files:
        print("No output files available for comparison.", flush=True)
        return 0

    if len(files) > 1:
        if not args.output_name:
            print("--output-name is not given, but found multiple outputs from model:")
            for f in files:
                print(f"  {os.path.basename(f)}")
            return 0
        first = os.path.join(args.output_dir, args.output_name)
        if not os.path.exists(first):
            print(f"{args.output_name} not found in {args.output_dir}", flush=True)
            return 1
    else:
        first = files[0]

    data = open(first, "rb").read()
    need = len(expected)
    actual = []
    for i in range(need):
        offset = i * 4
        if offset + 4 > len(data):
            break
        actual.append(struct.unpack_from("<f", data, offset)[0])

    print(
        f"Comparing first {len(actual)} values from {os.path.basename(first)} "
        f"against expected ({need} values requested):"
    )
    compared_result_txt = os.path.join(args.output_dir, "output_diff.txt")
    with open(compared_result_txt, "w") as fid:
        for idx, (exp, act) in enumerate(zip(expected, actual)):
            diff = abs(exp - act)
            rel = diff / (abs(exp) + 1e-9)
            status = "OK" if diff <= 1e-3 or rel <= 1e-3 else "DIFF"
            ret = (
                f"  idx {idx}: expected={exp:.6g}, actual={act:.6g}, "
                f"abs_diff={diff:.3g}, rel_diff={rel:.3g} -> {status}"
            )
            fid.write(ret + "\n")
            if idx < 10:
                print(ret)
            if idx == 10:
                print("  ...")
    print(f"Diff results saved in {compared_result_txt}")

    if len(actual) < need:
        print(f"Warning: output had only {len(actual)} float32 values, expected {need}.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
