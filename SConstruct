import os

cppSANN_version = 1.1
EIGEN_INCLUDE_PATH = "/usr/include/eigen3"

def printinfo(text):
	print("info: "+text)
	print("[cppSANN] Version - "+str(cppSANN_version))

if os.path.isfile(".sconsign.dblite"):
	os.remove(".sconsign.dblite")

MAIN_FILE = "test.cpp"
	
printinfo("cwd: "+os.getcwd())

project_dir = os.getcwd()
build_dir = os.path.join(project_dir,'build')
lib_dir = os.path.join(project_dir,'lib')
src_dir = os.path.join(project_dir,'src')


# Set our required libraries
libraries 		= []
library_paths 	= ''
include_paths	= [os.path.join(src_dir,'include'),EIGEN_INCLUDE_PATH]
cppDefines 		= {}
cppFlags 		= ['-Wall']#, '-Werror']
cxxFlags 		= ['-std=c++11']


#define environment

env = Environment()
env.Append(LIBS 			= libraries)
env.Append(LIBPATH 		= library_paths)
env.Append(CPPDEFINES 	= cppDefines)
env.Append(CPPFLAGS 		= cppFlags)
env.Append(CXXFLAGS 		= cxxFlags)
env.AppendUnique(CPPPATH = include_paths)

debug = ARGUMENTS.get('debug_info', 0)
if int(debug):
	print("[cppSANN] Debug Compilation")
	env.Append(CPPFLAGS = '-g')
else:
	print("[cppSANN] Release Compilation")
	env.Append(CPPFLAGS = '-O2')

SHARED_LIB_FLAG = False
shared_lib = ARGUMENTS.get('shared', 0)
if int(shared_lib):
	SHARED_LIB_FLAG = True
	print("[cppSANN] Compiling cppSANN to shared library")
	print("[cppSANN] .so will be saved to lib directory")

help_menu = ARGUMENTS.get('help', 0)
if int(help_menu):
	print("\n         Help Menu:\n------------------------------")
	print("  <Option> = 1/0")
	print("    help        - This help menu but compilation won't start")
	print("    debug       - Compiles with debug information")
	print("    shared      - Compiles as a shared library")

else:

	cpp_files = dict()

	# get all cpp from source
	for subdir, dirnames, filenames in os.walk(src_dir):
		for filename in filenames:
			if filename.endswith("cpp") or filename.endswith("cc"):
				if subdir not in cpp_files:
					cpp_files[subdir] = []
				cpp_files[subdir].append(filename)
	objects_list = []



	for subdir in cpp_files:
		matches = cpp_files[subdir]
		if MAIN_FILE in matches and not SHARED_LIB_FLAG:
			matches.insert(0, matches.pop(matches.index(MAIN_FILE)))	
		elif MAIN_FILE in matches and SHARED_LIB_FLAG:
			matches.remove(MAIN_FILE)
		
		for elem in matches:
			objects_list.append(env.Object(os.path.join(subdir,elem)))

	if SHARED_LIB_FLAG:
		env.Library(os.path.join(lib_dir,"cppSANN.so"), objects_list)
		print("[cppSANN] so file has never been checked")
	else:
		env.Program(os.path.join(build_dir,"cppSANN_exec"), objects_list)
