setup.py
src/
    mypkg/
        __init__.py
        module.py
        data/
            tables.dat
            spoons.dat
            forks.dat

setup(...,
      packages=['mypkg'], # target_path_name
      package_dir={'mypkg': 'src/mypkg'}, # target_path: source_path
      package_data={'mypkg': ['data/*.dat']}, # target_path: source_data
      )

the installed package should be like:

mypkg/
    __init__.py
    *.dat
                        