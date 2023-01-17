from gisttools.gist import load_gist_file, combine_gists, load_dx
import mdtraj as md
import numpy as np
import argparse
from os.path import splitext, basename


def weighting_presets(method, radius):
    """Return presets for the given distance weighting method
    
    Returns a tuple (method, parameters_dict)
    """
    if method == "piecewise_linear":
        return ("piecewise_linear", {"constant": radius-3, "cutoff": radius})
    elif method == "gaussian":
        return ("gaussian", {"sigma": radius/3})
    elif method == "logistic":
        return ("logistic", {"k": 20/radius, "x0": radius*0.75})
    else:
        raise ValueError(f"Unknown weighting method: {method}")


def gist_projection():
    """Command-line interface to projection_mean.

    Parameters
    ----------
    None, but reads command-line arguments.  Use gist_projection -h for more
    information.

    Returns
    -------
    None

    """
    args = _parse_args_gist_projection()
    if args.ewwref is None:
        print("WARNING: You did not supply a reference value for Eww.")
    combined_grid = load_grids(
        gist_file=args.gistfile,
        dx_files=args.dxfiles,
        Eww_ref=args.ewwref,
        ffmt=args.fileformat,
        autodetect_refcol=args.autodetect_refcol,
    )
    pdb = md.load_pdb(args.pdb).remove_solvent()
    if args.strip_H:
        pdb = pdb.atom_slice(pdb.top.select('element != "H"'))
    combined_grid.struct = pdb
    projection = get_projection(
        combined_grid,
        pdb,
        args.columns,
        rmax=args.radius,
        per_residue=args.per_residue,
        nearest=args.nearest,
    )
    for col in args.columns:
        if len(args.columns) > 1:
            current_pdbname = "{0}_{1}.pdb".format(splitext(args.pdbout)[0], col)
        else:
            current_pdbname = args.pdbout
        # Clip by the lowest and highest numbers possible in PDB B-Factors
        current_data = np.clip(projection[col].values, -9.99, 99.99)
        pdb.save_pdb(current_pdbname, force_overwrite=True, bfactors=current_data)
    return None


def get_projection(
    grid,
    pdb,
    columns,
    rmax : float,
    per_residue : bool,
    nearest : bool
    ):
    """Return a dataframe with the appropriate per-atom projection values."""
    if per_residue:
        if nearest:
            raise ValueError('nearest and per_residue can not be combined.')
        resis = [atom.residue.index for atom in pdb.top.atoms]
        out = grid.projection_mean(rmax=rmax, columns=columns, residues=resis)
    else:
        if nearest:
            out = grid.projection_nearest(rmax=rmax, columns=columns)
        else:
            out = grid.projection_mean(rmax=rmax, columns=columns)
    return out


def load_grids(gist_file, dx_files, Eww_ref=None, ffmt=None, autodetect_refcol='g_O'):
    """Load a gist output file + one or more dx files, return a Grid object.
    
    dx files are treated as containing _norm columns.
    """
    gist_obj = load_gist_file(
        gist_file,
        eww_ref=Eww_ref,
        struct=None,
        format=ffmt,
        autodetect_refcol=autodetect_refcol
    )
    # combined = combine_gists(gist_objs)
    xyz = gist_obj.grid.xyz(np.arange(gist_obj.grid.n_voxels))
    for fname_type in dx_files:
        if fname_type.endswith(':dens'):
            coltype = 'dens'
        elif fname_type.endswith(':norm'):
            coltype = 'norm'
        elif fname_type.endswith(':total'):
            coltype = 'total'
        else:
            raise ValueError("Specify dx file normalization by appending :dens, :norm or :total")
        fname = fname_type[:-(len(coltype)+1)]
        dx = load_dx(fname, colname='dx')
        fbase = splitext(basename(fname))[0]
        print(fbase + '_' + coltype)
        gist_obj[fbase + '_' + coltype] = dx.interpolate(['dx'], xyz)['dx']
    return gist_obj


def _parse_args_gist_projection():
    """Parse arguments for gist_projection."""
    typical_cols = ["dTStrans", "dTStorient", "dTSsix", "Esw", "Eww", "Eall", "A"]
    parser = argparse.ArgumentParser(
        description="Project the content of GIST output files onto atomic coordinates."
    )
    parser.add_argument(
        "gistfile",
        type=str,
        help="GIST output file in table format."
    )
    parser.add_argument(
        "-p",
        "--pdb",
        type=str,
        required=True,
        help="PDB file to which coordinates should be projected.",
    )
    parser.add_argument(
        "-dx",
        "--dxfiles",
        type=str,
        nargs="+",
        default=[],
        help=(
            "Extra files in OpenDX format. Those files will be interpolated "
            "to match your GIST grid, and will be available as extra columns, "
            "so you can do gist_projection ... -dx test.dx -c test."
        ),
    )
    parser.add_argument(
        "-o",
        "--out_pdb",
        dest="pdbout",
        type=str,
        default="GIST_PROJ.pdb",
        help=(
            "Output PDB file. The projected values will be written to the "
            "B-Factor column. If multiple columns are selected, writes multiple "
            "files."
        ),
    )
    parser.add_argument(
        "-e",
        "--eww_ref",
        dest="ewwref",
        type=float,
        help="Reference value for Eww-norm.",
    )
    parser.add_argument(
        "-r",
        "--radius",
        type=float,
        default=5.0,
        help="Radius for Schauperl projection.",
    )
    parser.add_argument(
        "-c",
        "--columns",
        type=str,
        nargs="+",
        default=["A"],
        help=(
            "GIST file column to be projected. For water using gigist, typical columns are "
            + ", ".join(typical_cols)
            + ". Not all fields have been tested though."
        ),
    )
    parser.add_argument(
        "--per_residue",
        action="store_true",
        help="Average quantities around whole residues, not around each object.",
    )
    parser.add_argument(
        "--strip_H",
        action="store_true",
        help="Whether to strip hydrogen from the PDB file.",
    )
    parser.add_argument(
        "--ffmt",
        dest="fileformat",
        type=str,
        default=None,
        help=(
            "Format of the GIST output files. Available "
            "options are: 'amber14', 'amber16', 'gigist'"
        ),
    )
    parser.add_argument(
        "--nearest",
        action="store_true",
        help="Use projection_nearest instead of projection_mean.",
    )
    parser.add_argument(
        "--autodetect_refcol",
        default="g_O",
        help="Name of reference column for detecting rho0 and num_frames. Set to g_C for chloroform.",
    )
    args = parser.parse_args()
    return args
