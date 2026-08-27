"""Validate a reference-training zip that came back from Kaggle.

Runs locally. The zip was produced on a machine we did not watch, by a person
who may not know what a retain-only reference is for, so nothing in it is
trusted until checked: not the file names, not the metadata, and certainly not
the claim that the model never saw the forget class. The last one is verified by
running the model, not by reading its sidecar.

Checks, per class found in the zip
----------------------------------
1. the zip extracts cleanly
2. the ``_best_dr`` checkpoint is present
3. it loads into our ResNet-18 CIFAR variant with ``strict=True`` -- which is
   also the architecture check: a different architecture fails to load
4. its metadata names the class we expected, by id **and** by name
5. the split sizes are 5 000 / 45 000 / 1 000 / 9 000
6. measured ``D_f_test`` accuracy is LOW  (the model cannot classify the class
   it never saw)
7. measured ``D_r_test`` accuracy is HIGH (it is still a good CIFAR-9 model)
8. the training log is present and its last epoch matches the configured budget

6 and 7 are the ones that matter. A reference that scores well on ``D_f_test``
was trained on the wrong split, and every objective computed against it would be
quietly wrong -- ``f1`` is a divergence *from this model*.

Writes ``results/reference_training/reference_validation_summary.csv`` and
prints a PASS/FAIL report. Exit code is non-zero if any class fails.

Run::

    python kaggle/reference_training/validate_reference_zip.py \\
        --zip reference_outputs_trial_ship.zip --expect-class 8
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402
from torch.utils.data import Subset  # noqa: E402

from medus_class.data import (  # noqa: E402
    CIFAR10_CLASS_NAMES,
    build_class_split,
    load_cifar10,
    make_loader,
)
from medus_class.evaluation.metrics import evaluate  # noqa: E402
from medus_class.models import build_model, load_checkpoint  # noqa: E402
from medus_class.utils.config import load_config, resolve_path  # noqa: E402
from medus_class.utils.device import get_device  # noqa: E402

EXPECTED_SPLIT_SIZES = (5_000, 45_000, 1_000, 9_000)

#: A retain-only reference must be close to useless on the class it never saw.
#: The finished frog reference measures 0.0000; anything above a few percent
#: means forget-class images leaked into training.
MAX_FORGET_TEST_ACC = 0.05

#: ...and still a good model on the other nine. The frog reference measures
#: 0.9459. This threshold is deliberately loose: it catches "training collapsed",
#: not "two tenths of a point below the frog run".
MIN_RETAIN_TEST_ACC = 0.90

CHECKPOINT_RE = re.compile(r"^class(\d)_([a-z]+)_reference_best_dr\.pt$")

SUMMARY_COLUMNS = [
    "class_id", "class_name", "verdict", "failures",
    "forget_test_acc", "forget_test_loss",
    "retain_test_acc", "retain_test_loss", "full_test_acc",
    "metadata_forget_class", "metadata_epoch", "metadata_seed",
    "d_f_train", "d_r_train", "d_f_test", "d_r_test",
    "log_epochs", "checkpoint_bytes", "source_zip",
]


def validate_one(checkpoint: Path, extracted: Path, cfg: dict[str, Any],
                 device: str, bundle: Any, source_zip: str) -> dict[str, Any]:
    """Every check for one class. Returns a summary row."""
    match = CHECKPOINT_RE.match(checkpoint.name)
    class_id, name = int(match.group(1)), match.group(2)
    failures: list[str] = []

    row: dict[str, Any] = {column: "" for column in SUMMARY_COLUMNS}
    row.update({"class_id": class_id, "class_name": name,
                "source_zip": source_zip,
                "checkpoint_bytes": checkpoint.stat().st_size})

    print(f"\n  class {class_id} ({name})")
    print(f"    checkpoint          {checkpoint.name} "
          f"({checkpoint.stat().st_size:,} bytes)")

    if CIFAR10_CLASS_NAMES[class_id] != name:
        failures.append(f"file name says class {class_id} is '{name}', "
                        f"but it is '{CIFAR10_CLASS_NAMES[class_id]}'")

    # --- load: this is also the architecture check ------------------------
    model = build_model(cfg["model"], num_classes=int(cfg["data"]["num_classes"]))
    try:
        metadata = load_checkpoint(checkpoint, model, map_location="cpu", strict=True)
        print(f"    loads into ResNet-18 CIFAR variant   OK (strict=True)")
    except Exception as exc:  # noqa: BLE001
        failures.append(f"checkpoint does not load: {type(exc).__name__}: {exc}")
        row["verdict"] = "FAIL"
        row["failures"] = " | ".join(failures)
        print(f"    LOAD FAILED: {exc}")
        return row

    metrics = metadata.get("metrics", {}) or {}
    row["metadata_forget_class"] = metrics.get("forget_class", "")
    row["metadata_epoch"] = metadata.get("epoch", "")
    row["metadata_seed"] = metadata.get("seed", "")

    recorded = metrics.get("forget_class")
    if recorded is None:
        failures.append("metadata has no forget_class")
    elif int(recorded) != class_id:
        failures.append(f"metadata forget_class is {int(recorded)}, "
                        f"file name says {class_id}")
    notes = metadata.get("notes") or ""
    if name not in notes:
        failures.append(f"metadata notes do not mention '{name}'")
    print(f"    metadata forget class  {recorded} ({name})   "
          f"epoch {metadata.get('epoch')}  seed {metadata.get('seed')}")

    # --- the split --------------------------------------------------------
    split = build_class_split(bundle.train_labels, bundle.test_labels, class_id)
    sizes = (int(split.forget_train.size), int(split.retain_train.size),
             int(split.forget_test.size), int(split.retain_test.size))
    row.update(dict(zip(("d_f_train", "d_r_train", "d_f_test", "d_r_test"), sizes)))
    if sizes != EXPECTED_SPLIT_SIZES:
        failures.append(f"split sizes {sizes} != {EXPECTED_SPLIT_SIZES}")
    print(f"    split sizes         {sizes}   "
          f"{'OK' if sizes == EXPECTED_SPLIT_SIZES else 'WRONG'}")

    # --- measure it -------------------------------------------------------
    eval_cfg = {**cfg["data"], "num_workers": 0, "persistent_workers": False}
    batch = int(cfg["data"]["batch_size"]["eval"])
    model.to(device).eval()

    forget_test = evaluate(model, make_loader(
        Subset(bundle.test, split.forget_test.tolist()), batch, False, eval_cfg, 42
    ), device, collect_per_sample=False)
    retain_test = evaluate(model, make_loader(
        Subset(bundle.test, split.retain_test.tolist()), batch, False, eval_cfg, 42
    ), device, collect_per_sample=False)

    n_f, n_r = forget_test.n_samples, retain_test.n_samples
    full_test_acc = (n_f * forget_test.accuracy + n_r * retain_test.accuracy) / (n_f + n_r)

    row.update({
        "forget_test_acc": round(forget_test.accuracy, 6),
        "forget_test_loss": round(forget_test.loss, 6),
        "retain_test_acc": round(retain_test.accuracy, 6),
        "retain_test_loss": round(retain_test.loss, 6),
        "full_test_acc": round(full_test_acc, 6),
    })

    if forget_test.accuracy > MAX_FORGET_TEST_ACC:
        failures.append(
            f"D_f_test accuracy {forget_test.accuracy:.4f} > "
            f"{MAX_FORGET_TEST_ACC} -- this model can classify the class it was "
            f"supposed never to have seen"
        )
    if retain_test.accuracy < MIN_RETAIN_TEST_ACC:
        failures.append(
            f"D_r_test accuracy {retain_test.accuracy:.4f} < "
            f"{MIN_RETAIN_TEST_ACC} -- training did not converge"
        )
    print(f"    D_f_test accuracy   {forget_test.accuracy:.4f}   "
          f"(must be <= {MAX_FORGET_TEST_ACC})   "
          f"{'OK' if forget_test.accuracy <= MAX_FORGET_TEST_ACC else 'FAIL'}")
    print(f"    D_r_test accuracy   {retain_test.accuracy:.4f}   "
          f"(must be >= {MIN_RETAIN_TEST_ACC})   "
          f"{'OK' if retain_test.accuracy >= MIN_RETAIN_TEST_ACC else 'FAIL'}")

    # --- the log ----------------------------------------------------------
    log_path = extracted / f"class{class_id}_{name}_training_log.csv"
    if not log_path.is_file():
        failures.append(f"missing {log_path.name}")
    else:
        with log_path.open(encoding="utf-8-sig") as handle:
            log_rows = list(csv.DictReader(handle))
        row["log_epochs"] = len(log_rows)
        expected_epochs = int(cfg["training"]["epochs"])
        if len(log_rows) != expected_epochs:
            failures.append(f"log has {len(log_rows)} epochs, config says "
                            f"{expected_epochs}")
        print(f"    training log        {len(log_rows)} epochs")

    for companion in (f"class{class_id}_{name}_reference_best_dr.json",
                      f"class{class_id}_{name}_training_summary.md"):
        if not (extracted / companion).is_file():
            failures.append(f"missing {companion}")

    row["verdict"] = "FAIL" if failures else "PASS"
    row["failures"] = " | ".join(failures)
    print(f"    VERDICT             {row['verdict']}")
    for line in failures:
        print(f"      - {line}")
    return row


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", required=True, help="the downloaded Kaggle zip")
    parser.add_argument("--expect-class", type=int, action="append", default=None,
                        help="class id that must be present; repeatable")
    parser.add_argument("--out",
                        default="results/reference_training/"
                                "reference_validation_summary.csv")
    parser.add_argument("--keep-extracted", default=None,
                        help="extract here instead of a temp directory")
    args = parser.parse_args()

    zip_path = Path(args.zip)
    if not zip_path.is_absolute():
        zip_path = (Path.cwd() / zip_path).resolve()
    if not zip_path.is_file():
        raise SystemExit(f"zip not found: {zip_path}")

    print("=" * 100)
    print(f"VALIDATING  {zip_path.name}  ({zip_path.stat().st_size:,} bytes)")
    print("=" * 100)

    temp: tempfile.TemporaryDirectory | None = None
    if args.keep_extracted:
        extracted = resolve_path(args.keep_extracted)
        extracted.mkdir(parents=True, exist_ok=True)
    else:
        temp = tempfile.TemporaryDirectory()
        extracted = Path(temp.name)

    try:
        with zipfile.ZipFile(zip_path) as archive:
            bad = archive.testzip()
            if bad is not None:
                raise SystemExit(f"zip is corrupt at member: {bad}")
            archive.extractall(extracted)
        print(f"  extracted cleanly to {extracted}")

        checkpoints = sorted(p for p in extracted.rglob("*.pt")
                             if CHECKPOINT_RE.match(p.name))
        if not checkpoints:
            raise SystemExit(
                "no *_reference_best_dr.pt found in the zip. Expected names "
                "like class8_ship_reference_best_dr.pt"
            )
        print(f"  found {len(checkpoints)} reference checkpoint(s): "
              + ", ".join(p.name for p in checkpoints))

        cfg = load_config("base.yaml")
        cfg.update(load_config("data/cifar10_class.yaml"))
        cfg.update(load_config("model/resnet18.yaml"))
        device = get_device(prefer=cfg["device"]["prefer"],
                            index=cfg["device"]["index"]).device
        print(f"  device {device}")
        bundle = load_cifar10({**cfg["data"], "download": False})

        rows = [validate_one(path, path.parent, cfg, device, bundle, zip_path.name)
                for path in checkpoints]

        missing: list[int] = []
        if args.expect_class:
            found = {int(r["class_id"]) for r in rows}
            missing = [c for c in args.expect_class if c not in found]
            for class_id in missing:
                print(f"\n  EXPECTED class {class_id} "
                      f"({CIFAR10_CLASS_NAMES[class_id]}) is NOT in this zip")

        out_path = resolve_path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Append across runs so a later class does not erase an earlier verdict.
        write_header = not out_path.is_file()
        with out_path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS)
            if write_header:
                writer.writeheader()
            writer.writerows(rows)

        failed = [r for r in rows if r["verdict"] == "FAIL"]
        print("\n" + "=" * 100)
        if failed or missing:
            print(f"RESULT: FAIL -- {len(failed)} of {len(rows)} class(es) failed"
                  + (f", {len(missing)} expected class(es) missing" if missing else ""))
        else:
            print(f"RESULT: PASS -- all {len(rows)} class(es) validated")
        print(f"  summary written to {out_path}")
        print("=" * 100)
        return 1 if (failed or missing) else 0
    finally:
        if temp is not None:
            temp.cleanup()


if __name__ == "__main__":
    raise SystemExit(main())
