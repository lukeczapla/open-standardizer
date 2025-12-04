#!/usr/bin/env python3
"""
CLI tool to compare ChemAxon-style enhanced SMILES stereo annotations
(A/B assignments in {...}) against RDKit CIP / E,Z assignment
after parsing.

Typical input format (CSV):

    id,smiles
    123,"C[C@H](O)CC{A2=s;B1,3=e}"
    124,"CC/C=C\Cl{B1,2=e}"
    125,CCO{A0=r}

Or just:

    smiles
    "C[C@H](O)CC{A2=s}"
    "CC/C=C\Cl{B1,2=e}"

Usage:

    python -m open_standardizer.testing.stereo_diff_cli data.csv
    cat data.csv | python -m open_standardizer.testing.stereo_diff_cli --only-failures
"""

from __future__ import annotations

import argparse
import csv
import sys
from typing import Iterable, Tuple, Optional

# Adjust this import to match your layout:
# if testing/ is a package under open_standardizer, this is fine:
from open_standardizer.stereo_validation import (
    validate_cip_assignments_on_smiles,
    StereoValidationResult,
)


def _iter_rows(path: Optional[str]) -> Iterable[Tuple[str, str]]:
    """
    Yield (record_id, enhanced_smiles) from a CSV-style input.

    Rules:
      - If the row has 1 column, we treat it as (id=smiles, smiles).
      - If the row has >=2 columns, use (row[0], row[1]).
      - Quotes are handled by the csv module, so commas inside SMILES are OK.
    """
    if path:
        fh = open(path, "r", newline="")
    else:
        fh = sys.stdin

    with fh:
        reader = csv.reader(fh)
        for row in reader:
            if not row:
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


def _print_result(rec_id: str, smiles: str, result: StereoValidationResult) -> None:
    """
    Human-readable report for a single record.
    """
    status = "PASS" if result.ok() else "FAIL"

    print(f"{rec_id}\t{status}\t{smiles}")

    # Atom-level details
    for comp in result.atom_matches:
        print(f"  ATOM {comp.atom_index}: {comp.assignment} == {comp.cip_code}  [match]")
    for comp in result.atom_mismatches:
        print(f"  ATOM {comp.atom_index}: {comp.assignment} != {comp.cip_code}  [mismatch]")
    for comp in result.atom_missing_cip:
        print(f"  ATOM {comp.atom_index}: {comp.assignment}  [missing RDKit CIP]")
    for comp in result.atom_non_cip_assignments:
        print(f"  ATOM {comp.atom_index}: {comp.assignment}  [non-CIP code / ignored]")

    # Bond-level details
    for comp in result.bond_matches:
        print(
            f"  BOND ({comp.atom_index1},{comp.atom_index2}): "
            f"{comp.assignment} == {comp.cip_code}  [match]"
        )
    for comp in result.bond_mismatches:
        print(
            f"  BOND ({comp.atom_index1},{comp.atom_index2}): "
            f"{comp.assignment} != {comp.cip_code}  [mismatch]"
        )
    for comp in result.bond_missing_cip:
        print(
            f"  BOND ({comp.atom_index1},{comp.atom_index2}): "
            f"{comp.assignment}  [missing RDKit E/Z]"
        )
    for comp in result.bond_unknown_assignments:
        print(
            f"  BOND ({comp.atom_index1},{comp.atom_index2}): "
            f"{comp.assignment}  [{comp.status}]"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Validate ChemAxon-style enhanced SMILES stereo annotations "
            "against RDKit CIP/E,Z assignments."
        )
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Input CSV file (id,smiles). If omitted, read from stdin.",
    )
    parser.add_argument(
        "--only-failures",
        action="store_true",
        help="Only print records where at least one stereo mismatch occurs.",
    )
    parser.add_argument(
        "--index-base",
        type=int,
        default=0,
        help=(
            "Indexing base for A/B assignments. "
            "0 = ChemAxon 0-based (default), 1 = 1-based."
        ),
    )

    args = parser.parse_args(argv)

    any_fail = False

    for rec_id, smi in _iter_rows(args.input):
        vr = validate_cip_assignments_on_smiles(smi, index_base=args.index_base)
        if vr is None:
            # couldn't parse / no meta
            print(f"{rec_id}\tINVALID\t{smi}")
            continue

        if args.only_failures and vr.ok():
            continue

        _print_result(rec_id, smi, vr)
        if not vr.ok():
            any_fail = True

    return 1 if any_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
