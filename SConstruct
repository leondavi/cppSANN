from __future__ import print_function

# Global Python includes
import itertools
import os
import re
import shutil
import subprocess
import sys

from os import mkdir, environ
from os.path import abspath, basename, dirname, expanduser, normpath
from os.path import exists,  isdir, isfile
from os.path import join as joinpath, split as splitpath
from re import match

def print_info(text):
	print("[info] "+text)


env = Environment()

debug = ARGUMENTS.get('debug', 0)

if int(debug):
	print_info("cxx debug flag -g is on")
	env.Append(CCFLAGS = '-g')


SConscript(['src/SConscript'],variant_dir='build',exports={'env':env},duplicate = 0)
