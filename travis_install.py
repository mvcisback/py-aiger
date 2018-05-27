#!/usr/bin/env python

import contextlib
import os
import subprocess
import tarfile
import zipfile
from pathlib import Path

GET_ABC_CMD = "wget https://github.com/berkeley-abc/abc/archive/master.zip"
GET_AIGER_CMD = "wget http://fmv.jku.at/aiger/aiger-1.9.9.tar.gz"

EXTRACT_ABC_CMD = "unzip master.zip -d /tmp/abc"
EXTRACT_AIGER_CMD = "tar -xvf {}/aiger-1.9.9.tar.gz"

INSTALL_ABC_CMD = "cmake . && make"
INSTALL_AIGER_CMD = "./configure.sh && make"


# https://stackoverflow.com/questions/41742317/how-can-i-change-directory-with-python-pathlib
@contextlib.contextmanager
def working_directory(path):
    """Changes working directory and returns to previous on exit."""
    prev_cwd = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev_cwd)


def install_aiger(aiger_path):
    aiger_path.mkdir()
    with working_directory(aiger_path):
        subprocess.check_call(GET_AIGER_CMD, shell=True)

        with tarfile.open("aiger-1.9.9.tar.gz") as f:
            f.extractall()

    with working_directory(aiger_path / "aiger-1.9.9"):
        subprocess.check_call(INSTALL_AIGER_CMD, shell=True)


def install_abc(abc_path):
    abc_path.mkdir()
    with working_directory(abc_path):
        subprocess.check_call(GET_ABC_CMD, shell=True)

        with zipfile.ZipFile(abc_path / "master.zip", "r") as f:
            f.extractall()

    with working_directory(abc_path / 'abc-master'):
        subprocess.check_call(INSTALL_ABC_CMD, shell=True)


def main():
    home = Path(os.environ['HOME'])
    aiger_path = home / ".cache/aiger"
    abc_path = home / ".cache/abc"

    if not aiger_path.exists():
        install_aiger(aiger_path)

    if not abc_path.exists():
        install_abc(abc_path)


if __name__ == '__main__':
    main()
