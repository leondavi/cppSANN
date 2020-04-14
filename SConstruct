import os

def printinfo(text):
	print("info: "+text)

if os.path.isfile(".sconsign.dblite"):
	os.remove(".sconsign.dblite")
	
printinfo("cwd: "+os.getcwd())

project_dir = os.getcwd()
build_dir = os.path.join(project_dir,'build')
src_dir = os.path.join(project_dir,'src')


# Set our required libraries
libraries 		= []
library_paths 	= ''
include_paths	= [os.path.join(src_dir,'include')]
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
	env.Append(CPPFLAGS = '-g')

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
	if "main.cpp" in matches:
		matches.insert(0, matches.pop(matches.index("main.cpp")))	
	
	for elem in matches:
		objects_list.append(env.Object(os.path.join(subdir,elem)))

env.Program(os.path.join(build_dir,"cppSANN_exec"), objects_list)

