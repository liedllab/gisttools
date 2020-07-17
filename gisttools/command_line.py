from gisttools.gist import loadGistFile, combine_gists, load_dx
import mdtraj as md
import numpy as np
import argparse
from os.path import splitext


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
        gist_files=args.gistfiles,
        dx_files=args.dxfiles,
        Eww_ref=args.ewwref,
        ffmt=args.fileformat,
        autodetect_refcol=args.autodetect_refcol,
    )
    pdb = md.load_pdb(args.pdb).remove_solvent()
    if args.strip_H:
        pdb = pdb.atom_slice(pdb.top.select('element != "H"'))
    combined_grid.struct = pdb

    if args.per_residue:
        if args.nearest:
            raise ValueError('nearest and per_residue can not be combined.')
        resis = [atom.residue.index for atom in pdb.top.atoms]
        proj_out = combined_grid.projection_mean(
            rmax=args.radius,
            columns=args.columns,
            residues=resis
        )
    else:
        if args.nearest:
            proj_out = combined_grid.projection_nearest(
                rmax=args.radius,
                columns=args.columns,
            )
        else:
            proj_out = combined_grid.projection_mean(
                rmax=args.radius,
                columns=args.columns,
            )
    for col in args.columns:
        if len(args.columns) > 1:
            current_pdbname = "{0}_{1}.pdb".format(splitext(args.pdbout)[0], col)
        else:
            current_pdbname = args.pdbout
        current_data = proj_out[col]
        cut_min, cut_max = -9.99, 99.99  # Maximum values for PDB B-Factors.
        current_data = np.maximum(current_data, cut_min)
        current_data = np.minimum(current_data, cut_max)
        pdb.save_pdb(current_pdbname, force_overwrite=True, bfactors=current_data)
    return None


def load_grids(gist_files, dx_files, Eww_ref=None, ffmt=None, autodetect_refcol='g_O'):
    gist_objs = [
        loadGistFile(fname, eww_ref=Eww_ref, struct=None, format=ffmt, autodetect_refcol=autodetect_refcol)
        for fname in gist_files
    ]
    combined = combine_gists(gist_objs)
    combined_xyz = combined.grid.xyz(np.arange(combined.grid.n_voxels))
    for fname in dx_files:
        dx = load_dx(fname)
        basename = list(dx.data.keys())[0]
        combined[basename + '_norm'] = dx.interpolate([basename], combined_xyz)[basename]
        combined[basename + '_dens'] = combined._norm2dens(combined[basename + '_norm'])
    return combined


def _parse_args_gist_projection():
    """Parse arguments for gist_projection."""
    typical_cols = [
        "dTStrans", "dTStorient", "dTSsix", "Esw", "Eww", "A", "Dipole_x", "Dipole_y",
        "Dipole_z", "Dipole", "neighbor",
    ]
    typical_cols += ['Eall_dens', 'Eall_norm', 'dTSall_dens', 'dTSall_norm',
               'A_dens', 'A_norm']
    parser = argparse.ArgumentParser(description='Project the content of GIST '
                                     'output files onto atomic coordinates '
                                     'using the Schauperl algorithm.')
    parser.add_argument('gistfiles', type=str, nargs='+',
                        help='One or more GIST output files. If more than '
                        'one file is given, they will be combined to a single '
                        'grid.')
    parser.add_argument('-p', '--pdb', type=str, required=True,
                        help='PDB file to which '
                        'coordinates should be projected.')
    parser.add_argument('-dx', '--dxfiles', type=str, nargs='+', default=[],
                        help=('Extra files in OpenDX format. Those files will be '
                        'interpolated to match your GIST grid, and will be available as '
                        'extra columns, so you can do gist_projection ... -dx test.dx -c test.'))
    parser.add_argument('-o', '--out_pdb', dest='pdbout', type=str,
                        default='GIST_PROJ.pdb', help='Output PDB file. '
                        'The projected values will be written to the B-Factor '
                        'column. If multiple columns are selected, writes '
                        'multiple files.')
    parser.add_argument('-e', '--eww_ref', dest='ewwref', type=float,
                        help='Reference value for Eww-norm.')
    parser.add_argument('-r', '--radius', type=float, default=5.0,
                        help='Radius for Schauperl projection.')
    parser.add_argument('-c', '--columns', type=str, nargs='+', default=['A'],
                        help='GIST file column to be projected. For water using gigist, '
                        'typical columns are '+', '.join(typical_cols)+'. Not all fields '
                        'have been tested though.')
    parser.add_argument('--per_residue', action='store_true',
                        help='Average quantities around whole residues, not '
                        'around each object.')
    parser.add_argument('--strip_H', action='store_true',
                        help='Whether to strip hydrogen from the PDB file.')
    parser.add_argument('--ffmt', dest='fileformat', type=str, default=None,
                        help=('Format of the GIST output files. Available '
                              'options are: "amber14", "amber16", "gigist"'))
    parser.add_argument('--nearest', action='store_true',
                        help='Use projection_nearest instead of projection_mean.')
    parser.add_argument('--autodetect_refcol', default='g_O',
                        help='Name of reference column for detecting rho0 and num_frames. Set to g_C for chloroform.')
    args = parser.parse_args()
    return args
