from setuptools import setup, find_packages

setup(
    name="gisttools",
    version="0.3",
    description="Tools to process and visualize GIST results from cpptraj.",
    author="Franz Waibl",
    author_email="franz.waibl@uibk.ac.at",
    include_package_data=True,
    zip_safe=False,
    install_requires=['numpy', 'pandas', 'numba', 'mdtraj', 'scipy', 'GridDataFormats'],
    setup_requires=['pytest_runner'],
    tests_require=['pytest'],
    py_modules=[
        "gisttools.grid",
        "gisttools.utils",
        "gisttools.command_line",
        "gisttools.gist",
        "gisttools.shape_buffer",
    ],
    entry_points={
        "console_scripts": ["gist_projection=gisttools.command_line:gist_projection"]
    },
)
