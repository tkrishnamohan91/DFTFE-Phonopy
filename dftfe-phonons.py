#!/usr/bin/env python3
"""
dftfe-phonon: Phonopy <-> DFT-FE bridge (two commands, interactive prompts)

* --prepare_input :
    - Read every POSCAR-* in the current folder
    - Detect unique elements across all files
    - PROMPT for valence electrons for each element (atomic numbers auto-detected)
    - PROMPT for magnetism:
        • Is the calculation magnetic?
        • If yes, choose either a single starting magnetization for all elements
          OR per-element starting magnetizations
    - For each POSCAR-*, create dftfe_in/POSCAR-*/ with
        domainBoundingVectors.inp (Bohr) and coordinates.inp

* --read_force :
    - For each POSCAR-* (with matching dftfe_in/POSCAR-*/forces.txt),
      detect the displaced atom and vector (vs reference SPOSCAR or --reference)
    - Convert forces from the given unit to eV/Å (default: Ha/Bohr)
    - Write FORCE_SETS (Type-1) for Phonopy

Usage
-----
# Interactive (valence + magnetism prompts)
python dftfe_phonon.py --prepare_input

# Non-interactive magnetism (global value)
python dftfe_phonon.py --prepare_input --zvalence-json zvalence.json --magnetic --magvalue 2.0

# Assemble FORCE_SETS (forces in Ha/Bohr by default)
python dftfe_phonon.py --read_force --reference SPOSCAR --force-unit habohr

"""

import argparse
import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

from phonopy.interface.vasp import read_vasp

# ----------------- Periodic Table Utilities -----------------

PTABLE = [
    "H","He",
    "Li","Be","B","C","N","O","F","Ne",
    "Na","Mg","Al","Si","P","S","Cl","Ar",
    "K","Ca","Sc","Ti","V","Cr","Mn","Fe","Co","Ni","Cu","Zn",
    "Ga","Ge","As","Se","Br","Kr",
    "Rb","Sr","Y","Zr","Nb","Mo","Tc","Ru","Rh","Pd","Ag","Cd",
    "In","Sn","Sb","Te","I","Xe",
    "Cs","Ba","La","Ce","Pr","Nd","Pm","Sm","Eu","Gd","Tb","Dy",
    "Ho","Er","Tm","Yb","Lu",
    "Hf","Ta","W","Re","Os","Ir","Pt","Au","Hg",
    "Tl","Pb","Bi","Po","At","Rn","Fr","Ra","Ac","Th",
    "Pa","U","Np","Pu","Am","Cm","Bk","Cf","Es","Fm",
    "Md","No","Lr","Rf","Db","Sg","Bh","Hs","Mt","Ds",
    "Rg","Cn","Nh","Fl","Mc","Lv","Ts","Og"
]
SYMBOL_TO_Z: Dict[str, int] = {sym: (i + 1) for i, sym in enumerate(PTABLE)}

def symbol_to_Z(sym: str) -> int:
    s = sym.strip()
    if len(s) == 0:
        raise ValueError("Empty element symbol")
    s = s[0].upper() + s[1:].lower()
    if s not in SYMBOL_TO_Z:
        raise KeyError(f"Unknown element symbol '{sym}'")
    return SYMBOL_TO_Z[s]

# ----------------- General Utilities -----------------

def natural_key(s: str):
    """Natural sort key so POSCAR-2 < POSCAR-10."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", s)]

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def prompt_yes_no(msg: str, default: Optional[bool] = None) -> bool:
    """Interactive yes/no prompt with optional default."""
    while True:
        suffix = " [y/n]" if default is None else (" [Y/n]" if default else " [y/N]")
        ans = input(msg + suffix + ": ").strip().lower()
        if ans == "" and default is not None:
            return default
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Please answer y or n.")

def prompt_int(msg: str) -> int:
    while True:
        s = input(msg).strip()
        try:
            v = int(s)
            if v <= 0:
                print("Please enter a positive integer.")
                continue
            return v
        except ValueError:
            print("Please enter an integer (e.g., 8 for Fe, 4 for C).")

def prompt_float(msg: str) -> float:
    while True:
        s = input(msg).strip()
        try:
            return float(s)
        except ValueError:
            print("Please enter a number (e.g., 2.0).")


def get_symbols_from_atoms(atoms) -> List[str]:
    """Return list of element symbols from ASE Atoms or PhonopyAtoms.
    Compatible with phonopy versions where PhonopyAtoms does not expose
    get_chemical_symbols() like ASE.Atoms."""
    # ASE Atoms or newer PhonopyAtoms that implement get_chemical_symbols
    if hasattr(atoms, "get_chemical_symbols"):
        return list(atoms.get_chemical_symbols())
    # PhonopyAtoms usually has a "symbols" attribute (sequence-like)
    if hasattr(atoms, "symbols"):
        syms = atoms.symbols
        try:
            return list(syms)
        except TypeError:
            return [str(syms)]
    # Fallback: try atomic numbers if available
    if hasattr(atoms, "numbers"):
        return [PTABLE[int(Z) - 1] for Z in atoms.numbers]
    raise AttributeError("Cannot extract chemical symbols from atoms object")

# ----------------- Units -----------------
HARTREE_TO_EV = 27.211386245988
BOHR_TO_ANG   = 0.529177210903
ANG_TO_BOHR   = 1.0 / BOHR_TO_ANG  # 1 Å = 1.889726124565062 Bohr
HABOHR_TO_EVA = HARTREE_TO_EV / BOHR_TO_ANG  # Ha/Bohr -> eV/Å

def convert_forces(forces: np.ndarray, unit: str) -> np.ndarray:
    """Convert forces from given unit to eV/Å."""
    unit = unit.lower()
    if unit in ("ev/ang", "evang", "ev/a"):
        return forces
    if unit in ("ha/bohr", "habohr"):
        return forces * HABOHR_TO_EVA
    raise ValueError(f"Unknown force unit '{unit}'. Supported: 'eV/Ang', 'Ha/Bohr'.")

# ----------------- DFT-FE Input Writers -----------------

@dataclass
class ZValEntry:
    Z: int
    valence: int

def build_zval_map(all_elements: List[str], zvalence_json: Optional[str]) -> Dict[str, Tuple[int,int]]:
    """
    Return mapping element -> (Z, valence).
    If zvalence_json is given, read mapping from there (keys are symbols).
    Otherwise prompt interactively.
    """
    zval_map: Dict[str, Tuple[int,int]] = {}

    if zvalence_json is not None:
        with open(zvalence_json) as f:
            data = json.load(f)
        for sym, pair in data.items():
            if len(pair) != 2:
                raise ValueError(f"Zval for {sym} must be [Z, valence].")
            Z, val = int(pair[0]), int(pair[1])
            if Z != symbol_to_Z(sym):
                print(f"Warning: Z mismatch for {sym}: file says {Z}, table says {symbol_to_Z(sym)}.")
            zval_map[sym] = (Z, val)
        # ensure all elements present
        for sym in all_elements:
            if sym not in zval_map:
                raise KeyError(f"Element '{sym}' missing in zvalence_json.")
        return zval_map

    print("\n=== Valence configuration for each element ===")
    for sym in all_elements:
        print(f"\nElement: {sym}")
        Z = symbol_to_Z(sym)
        print(f"  Atomic number Z = {Z}")
        val = prompt_int("  Number of valence electrons to use in DFT-FE = ")
        zval_map[sym] = (Z, val)
    print("\n[Info] Valence mapping chosen:")
    for k, (Z, val) in zval_map.items():
        print(f"  {k}: Z={Z}, valence={val}")
    return zval_map

def write_dftfe_inputs(
    poscar_path: Path,
    out_dir: Path,
    zval_map: Dict[str, Tuple[int,int]],
    magnetic: bool,
    magvalue: Optional[float],
    mag_map: Optional[Dict[str,float]],
    vectors_filename: str = "domainBoundingVectors.inp",
    coords_filename: str   = "coordinates.inp",
):
    """Write DFT-FE domainBoundingVectors.inp and coordinates.inp."""
    atoms = read_vasp(str(poscar_path))  # PhonopyAtoms
    lat = np.array(atoms.cell, dtype=float)               # (3,3), Å
    frac = np.array(atoms.get_scaled_positions(), float)  # (N,3)
    symbols = get_symbols_from_atoms(atoms)

    ensure_dir(out_dir)

    # domainBoundingVectors.inp (DFT-FE expects Bohr)
    lat_bohr = lat * ANG_TO_BOHR
    with open(out_dir / vectors_filename, "w") as f:
        for row in lat_bohr:
            f.write("{:20.12f} {:20.12f} {:20.12f}\n".format(*row))

    # coordinates.inp
    with open(out_dir / coords_filename, "w") as f:
        for s, r in zip(symbols, frac):
            if s not in zval_map:
                raise KeyError(f"Element '{s}' missing in zvalence mapping.")
            Z, val = zval_map[s]
            if magnetic:
                m = (mag_map[s] if (mag_map is not None and s in mag_map) else magvalue)
                f.write(f"{Z:3d} {val:3d} {r[0]:20.12f} {r[1]:20.12f} {r[2]:20.12f} {m:12.6f}\n")
            else:
                f.write(f"{Z:3d} {val:3d} {r[0]:20.12f} {r[1]:20.12f} {r[2]:20.12f}\n")

# ----------------- Displacement Detection -----------------

def fractional_min_image(dfrac: np.ndarray) -> np.ndarray:
    """Wrap fractional deltas to [-0.5, 0.5) using nearest image."""
    return dfrac - np.round(dfrac)

def detect_displacement(
    frac_ref: np.ndarray,
    frac_disp: np.ndarray,
    tol: float = 1e-5
) -> Tuple[int, np.ndarray]:
    """
    Given reference and displaced fractional coords (N,3),
    find index of displaced atom and its displacement in Cartesian
    (still in fractional for now, caller multiplies by lattice).
    Returns (index, dfrac).
    """
    if frac_ref.shape != frac_disp.shape:
        raise ValueError("Reference and displaced structures have different shapes.")
    dfrac_all = fractional_min_image(frac_disp - frac_ref)
    norms = np.linalg.norm(dfrac_all, axis=1)
    idx = np.argmax(norms)
    max_norm = norms[idx]
    if max_norm < tol:
        raise ValueError("No significant displacement detected.")
    return idx, dfrac_all[idx]

# ----------------- FORCE_SETS Writer (Type-1) -----------------

def write_FORCE_SETS_type1(
    out_path: Path,
    entries: List[Tuple[int, np.ndarray, np.ndarray]],
    natoms: int
):
    """
    Write Phonopy FORCE_SETS (type 1).
    entries: list of (atom_index, displacement_cart[3], forces[N,3] in eV/Å).
    natoms: total number of atoms in supercell.
    """
    with open(out_path, "w") as f:
        f.write("1\n")  # format version
        f.write(f"{natoms} {len(entries)}\n")
        for (idx0, dcart, forces) in entries:
            f.write(f"{idx0 + 1}\n")  # 1-based index
            f.write("{:20.16f} {:20.16f} {:20.16f}\n".format(*dcart))
            for row in forces:
                f.write("{:20.16f} {:20.16f} {:20.16f}\n".format(*row))

# ----------------- Forces Reader -----------------

def read_forces_txt(force_file: Path, natoms: int, unit: str) -> np.ndarray:
    """
    Read forces from 'forces.txt' (DFT-FE output).
    Expected format: one atom per line, columns Fx Fy Fz (in given unit).
    """
    data = []
    with open(force_file) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) < 3:
                raise ValueError(f"Bad forces line in {force_file}: {line}")
            fx, fy, fz = map(float, parts[:3])
            data.append([fx, fy, fz])
    arr = np.array(data, float)
    if arr.shape != (natoms, 3):
        raise ValueError(f"Forces in {force_file} have shape {arr.shape}, expected ({natoms}, 3)")
    return convert_forces(arr, unit)

# ----------------- Magnetism Handling -----------------

def prompt_magnetism(all_elements: List[str]) -> Tuple[bool, Optional[float], Optional[Dict[str,float]]]:
    """
    Interactively ask if the user wants magnetism & how.
    Returns (magnetic, global_mag, per_element_map).
    If global_mag is not None, that value is applied to all elements
    unless overwritten by the per_element_map.
    """
    magnetic = prompt_yes_no("\nIs this a magnetic calculation?", default=False)
    if not magnetic:
        print("  -> No magnetism column will be written to coordinates.inp.")
        return False, None, None

    mode_global = prompt_yes_no("Use a single starting magnetization for ALL elements?", default=True)
    if mode_global:
        mag = prompt_float("  Starting magnetization (e.g., 2.0 for Fe) = ")
        print(f"  -> Will use m={mag} for all elements.")
        return True, mag, None

    # Per-element magnetization
    per_map: Dict[str,float] = {}
    print("\n=== Per-element starting magnetizations ===")
    for sym in all_elements:
        mag = prompt_float(f"  Starting magnetization for {sym} = ")
        per_map[sym] = mag
    print("\n[Info] Magnetization mapping chosen:")
    for k, m in per_map.items():
        print(f"  {k}: m={m}")
    return True, None, per_map

def noninteractive_magnetism(all_elements: List[str], magvalue: float) -> Tuple[bool, float, None]:
    """Non-interactive magnets: global value."""
    print("\n[Non-interactive magnetism]")
    print(f"  All elements will get starting magnetization = {magvalue}")
    return True, magvalue, None

# ----------------- Main CLI -----------------

def main():
    ap = argparse.ArgumentParser(description="dftfe-phonon utility")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare_input", action="store_true", help="Create DFT-FE inputs for POSCAR-*")
    g.add_argument("--read_force",   action="store_true", help="Read forces and write FORCE_SETS")

    ap.add_argument("--poscar-glob", default="POSCAR-*", help="Glob for displaced supercells")
    ap.add_argument("--out-root", default="dftfe_in", help="Root folder to place per-POSCAR outputs")
    ap.add_argument("--vectors-filename", default="domainBoundingVectors.inp",
                    help="Filename for lattice vectors file (DFT-FE expects Bohr)")
    ap.add_argument("--coords-filename", default="coordinates.inp",
                    help="Filename for coordinates file")

    # Z/valence & magnetism (prepare_input)
    ap.add_argument("--zvalence-json",
                    help="Optional JSON mapping like {'Fe':[26,8],'C':[6,4]} to skip valence prompts")
    ap.add_argument("--magnetic", action="store_true",
                    help="Non-interactive: add starting magnetization column (global)")
    ap.add_argument("--magvalue", type=float, default=None,
                    help="Non-interactive: global starting magnetization value")

    # read_force options
    ap.add_argument("--forces-filename", default="forces.txt", help="Per-folder forces file")
    ap.add_argument("--reference", help="Reference supercell POSCAR (e.g., SPOSCAR)")
    ap.add_argument("--output", default="FORCE_SETS", help="Output FORCE_SETS filename")
    ap.add_argument("--force-unit", default="Ha/Bohr",
                    help="Unit of forces in forces.txt (Ha/Bohr or eV/Ang).")

    args = ap.parse_args()

    if args.prepare_input:
        # Find all POSCAR-* files
        poscars = sorted(Path(".").glob(args.poscar_glob), key=lambda p: natural_key(p.name))
        if not poscars:
            raise SystemExit(f"No files matching {args.poscar_glob} in current directory.")

        print(f"\nFound {len(poscars)} POSCAR files:")
        for p in poscars:
            print(f"  {p}")

        # Collect all elements
        all_elements: List[str] = []
        for poscar in poscars:
            atoms = read_vasp(str(poscar))
            all_elements.extend(get_symbols_from_atoms(atoms))

        all_elements = sorted(set(all_elements), key=lambda s: SYMBOL_TO_Z[s])
        print("\nUnique elements detected:", ", ".join(all_elements))

        # Valence mapping
        zval_map = build_zval_map(all_elements, args.zvalence_json)

        # Magnetism
        if args.magnetic:
            if args.magvalue is None:
                raise SystemExit("--magnetic requires --magvalue for non-interactive mode.")
            magnetic, magvalue, mag_map = noninteractive_magnetism(all_elements, args.magvalue)
        else:
            magnetic, magvalue, mag_map = prompt_magnetism(all_elements)

        # For each POSCAR, write DFT-FE inputs
        for poscar in poscars:
            name = poscar.name
            out_dir = Path(args.out_root) / name
            print(f"\n[prepare_input] Writing DFT-FE inputs for {name} -> {out_dir}")
            write_dftfe_inputs(
                poscar_path=poscar,
                out_dir=out_dir,
                zval_map=zval_map,
                magnetic=magnetic,
                magvalue=magvalue,
                mag_map=mag_map,
                vectors_filename=args.vectors_filename,
                coords_filename=args.coords_filename,
            )

        print("\n[prepare_input] Done.")
        return

    # --- read_force mode ---

    if args.reference is None:
        raise SystemExit("--read_force requires --reference POSCAR (e.g., SPOSCAR).")

    ref_atoms = read_vasp(args.reference)
    lat_ref = np.array(ref_atoms.cell, float)
    frac_ref = np.array(ref_atoms.get_scaled_positions(), float)
    natoms_ref = frac_ref.shape[0]

    # Displaced structures
    poscars = sorted(Path(".").glob(args.poscar_glob), key=lambda p: natural_key(p.name))
    poscars = [p for p in poscars if p.name != os.path.basename(args.reference)]
    if not poscars:
        raise SystemExit(f"No displaced POSCAR-* found (glob={args.poscar_glob}).")

    entries: List[Tuple[int,np.ndarray,np.ndarray]] = []

    for poscar in poscars:
        disp_atoms = read_vasp(str(poscar))
        lat = np.array(disp_atoms.cell, float)
        frac_disp = np.array(disp_atoms.get_scaled_positions(), float)
        if frac_disp.shape != frac_ref.shape:
            raise SystemExit(f"{poscar} has different number of atoms vs reference.")

        # Detect which atom is displaced and by how much (in fractional coords)
        idx0, dfrac = detect_displacement(frac_ref, frac_disp)
        dcart = lat_ref.T @ dfrac  # use reference lattice for displacement vector

        # Read forces
        folder = Path(args.out_root) / poscar.name
        force_file = folder / args.forces_filename
        if not force_file.exists():
            raise SystemExit(f"Missing forces file: {force_file}")
        forces = read_forces_txt(force_file, natoms_ref, args.force_unit)

        entries.append((idx0, dcart, forces))
        print(f"[read_force] {poscar.name}: atom #{idx0+1}, |disp|={np.linalg.norm(dcart):.5e} Å, forces OK")

    # 3) write FORCE_SETS
    write_FORCE_SETS_type1(Path(args.output), entries, natoms_ref)
    print(f"[read_force] Wrote {args.output} with {len(entries)} displacements.")

if __name__ == "__main__":
    main()

