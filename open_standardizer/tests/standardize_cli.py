#!/usr/bin/env python
"""
CLI wrapper around open_standardizer's standardization pipeline.

Features
--------
- Accepts ChemAxon-style *enhanced SMILES*:
      "c1ccccc1[NH2+] {A7=s;A9=r;o1:7,9}"
  or plain SMILES.

- Can run either:
    * XML-driven pipeline (ChemAxon-style XML config)
    * Or the DEFAULT_OPS-based pipeline (clear_stereo → ... → aromatize)

- Handles curly metadata the same way your Python API does:
    * default: preserve original { ... } block
    * --regenerate-stereo: rebuild stereo curly block from RDKit (CIP/groups),
      optionally merging back other tokens
    * --drop-meta: strip metadata completely, return plain SMILES

Input / Output
--------------
Input: CSV on stdin or from a file, with either

    id,smiles
    123,"C[C@H](O)CC {A2=s;B1,3=e}"
    124,"CC/C=C\\Cl {B1,2=e}"

or just

    smiles
    "C[C@H](O)CC {A2=s}"
    "CC/C=C\\Cl {B1,2=e}"

Output: CSV to stdout:

    id,orig_smiles,std_smiles,status

where status is "OK" or "ERROR".
"""

from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor
from typing import Iterable, Optional, Tuple, List

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.error")
RDLogger.DisableLog("rdApp.warning")

from open_standardizer.standardize import (
    standardize,
    standardize_smiles,
    standardize_any,
    DEFAULT_OPS,
)
from open_standardizer.gpu_cpu_policy_manager import GPU_CPU_MANAGER, Policy


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _core_smiles(s: str) -> str:
    """
    Strip ChemAxon-style curly metadata for display only.

    Example:
      'CCC[NH3+] {A7=s;B1,3=e}' -> 'CCC[NH3+]'
      'CCO'                      -> 'CCO'
    """
    return s.split(" {", 1)[0]


def _iter_rows(path: Optional[str]) -> Iterable[Tuple[str, str]]:
    """
    Yield (record_id, smiles) from a CSV-style input.

    Rules:
      - If the row has 1 column, treat it as (id=smiles, smiles).
      - If the row has >=2 columns, use (row[0], row[1]).
      - Quotes are handled by the csv module, so commas inside SMILES are OK.
    """
    if path:
        fh = open(path, "r", newline="")
    else:
        fh = sys.stdin

    with fh:
        reader = csv.reader(fh)
        header_peeked = False

        for row in reader:
            if not row:
                continue

            # Heuristic header skip if "smiles" appears
            if not header_peeked:
                header_peeked = True
                if any(col.lower() == "smiles" for col in row):
                    continue

            if len(row) == 1:
                smi = row[0].strip()
                if not smi:
                    continue
                yield (smi, smi)
            else:
                rec_id = row[0].strip()
                smi = row[1].strip()
                if not smi:
                    continue
                if not rec_id:
                    rec_id = smi
                yield (rec_id, smi)


# ---------------------------------------------------------------------------
# Core standardization helpers
# ---------------------------------------------------------------------------

def _standardize_with_xml(
    smiles: str,
    xml_path: str,
    preserve_chemaxon_meta: bool,
) -> Optional[str]:
    """
    XML-driven pipeline. Uses standardize_smiles, which is already
    enhanced-SMILES-aware and runs through Standardizer.standardize().
    """
    return standardize_smiles(
        smiles=smiles,
        xml_path=xml_path,
        preserve_chemaxon_meta=preserve_chemaxon_meta,
    )


def _standardize_with_ops(
    smiles: str,
    ops: List[str],
    policy: Policy,
    preserve_chemaxon_meta: bool,
    regenerate_chemaxon_meta: bool,
    index_base: int,
) -> Optional[str]:
    """
    DEFAULT_OPS / custom ops pipeline, enhanced-SMILES-aware.

    Uses the Mol-based pipeline underneath via standardize_any, so we
    only do SMILES→Mol once and Mol→SMILES once.
    """
    # If we are dropping all metadata, we can just ask for plain SMILES.
    if not preserve_chemaxon_meta and not regenerate_chemaxon_meta:
        return standardize_any(
            smiles,
            ops=ops,
            policy=policy,
            assume_enhanced_smiles=True,
            index_base=index_base,
            output_format="smiles",
        )

    # If we want to preserve or regenerate ChemAxon meta, use the high-level
    # enhanced-SMILES helper.
    return standardize(
        smiles=smiles,
        ops=ops,
        policy=policy,
        preserve_chemaxon_meta=preserve_chemaxon_meta,
        regenerate_chemaxon_meta=regenerate_chemaxon_meta,
        index_base=index_base,
    )


def _process_smiles(
    smiles: str,
    args: argparse.Namespace,
) -> Tuple[str, str]:
    orig = smiles

    try:
        if args.drop_meta:
            preserve_meta = False
            regenerate_stereo = False
        else:
            preserve_meta = True
            regenerate_stereo = bool(args.regenerate_stereo)

        if args.xml_config:
            out = _standardize_with_xml(
                smiles=smiles,
                xml_path=args.xml_config,
                preserve_chemaxon_meta=preserve_meta,
            )
        else:
            ops = args.ops or list(DEFAULT_OPS)
            out = _standardize_with_ops(
                smiles=smiles,
                ops=ops,
                policy=GPU_CPU_MANAGER,
                preserve_chemaxon_meta=preserve_meta,
                regenerate_chemaxon_meta=regenerate_stereo,
                index_base=args.index_base,
            )

        if out is None:
            return orig, "ERROR"
        return out, "OK"

    except KeyboardInterrupt:
        raise
    except Exception as e:
        sys.stderr.write(f"[ERROR] {e!r} while processing: {orig}\n")
        return orig, "ERROR"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _worker_task(args_tuple):
    rec_id, smi, args = args_tuple
    std_smi, status = _process_smiles(smi, args)
    return rec_id, smi, std_smi, status


def _update_progress(done: int, total: int):
    """
    Simple stderr progress: "Processed X/Y".
    Called from the main process only.
    """
    if total <= 0:
        return
    # update every 100 molecules, and at the very end
    if (done % 100 == 0) or (done == total):
        sys.stderr.write(f"\rProcessed {done}/{total} molecules")
        sys.stderr.flush()


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Standardize SMILES / enhanced SMILES using open_standardizer.\n"
            "Reads CSV (id,smiles) or single-column SMILES from stdin or a file,\n"
            "writes CSV with id,orig_smiles,std_smiles,status."
        )
    )

    parser.add_argument(
        "input",
        nargs="?",
        help="Input CSV file (id,smiles). If omitted, read from stdin.",
    )
    parser.add_argument(
        "--xml-config",
        help=(
            "Path to ChemAxon-style XML config. "
            "If provided, the XML pipeline is used instead of DEFAULT_OPS."
        ),
    )
    parser.add_argument(
        "--ops",
        nargs="+",
        help=(
            "Explicit list of ops to run (overrides DEFAULT_OPS). "
            "Ignored if --xml-config is given. "
            "Example: --ops clear_stereo neutralize aromatize"
        ),
    )
    parser.add_argument(
        "--index-base",
        type=int,
        default=0,
        help=(
            "Indexing base for enhanced SMILES metadata (A/B/group indices). "
            "0 = ChemAxon 0-based (default), 1 = 1-based."
        ),
    )
    parser.add_argument(
        "--regenerate-stereo",
        action="store_true",
        help=(
            "After standardization, regenerate the stereo curly block "
            "from RDKit CIP/groups (A/B/B tokens). Non-stereo tokens are "
            "preserved if present in the original metadata."
        ),
    )
    parser.add_argument(
        "--drop-meta",
        action="store_true",
        help=(
            "Drop any ChemAxon curly metadata entirely and return plain SMILES. "
            "Overrides --regenerate-stereo."
        ),
    )
    parser.add_argument(
        "--only-errors",
        action="store_true",
        help=(
            "Only emit rows where status == ERROR. "
            "Useful for triage runs on large libraries."
        ),
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of parallel workers (processes). 1 = no parallelism.",
    )

    args = parser.parse_args(argv)

    writer = csv.writer(sys.stdout)
    writer.writerow(["id", "orig_smiles", "std_smiles", "status"])

    # materialize all rows so we know total for progress
    rows = list(_iter_rows(args.input))
    total = len(rows)
    any_error = False

    try:
        if args.workers <= 1:
            # single-process path
            for i, (rec_id, smi) in enumerate(rows, start=1):
                std_smi, status = _process_smiles(smi, args)

                if args.only_errors and status == "OK":
                    _update_progress(i, total)
                    continue

                orig_core = _core_smiles(smi)
                std_core = _core_smiles(std_smi)
                writer.writerow([rec_id, orig_core, std_core, status])
                if status != "OK":
                    any_error = True

                _update_progress(i, total)
        else:
            # multi-process path
            items = [(rec_id, smi, args) for (rec_id, smi) in rows]

            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                # map preserves order; we just wrap the iterator with an index
                for i, (rec_id, smi, std_smi, status) in enumerate(
                    ex.map(_worker_task, items), start=1
                ):
                    if args.only_errors and status == "OK":
                        _update_progress(i, total)
                        continue
                    orig_core = _core_smiles(smi)
                    std_core = _core_smiles(std_smi)

                    writer.writerow([rec_id, orig_core, std_core, status])
                    if status != "OK":
                        any_error = True

                    _update_progress(i, total)

    except KeyboardInterrupt:
        sys.stderr.write("\n[INFO] Interrupted by user (Ctrl+C)\n")
        sys.stderr.flush()
        return 130
    finally:
        if total > 0:
            sys.stderr.write("\n")
            sys.stderr.flush()

    return 1 if any_error else 0


if __name__ == "__main__":
    raise SystemExit(main())
