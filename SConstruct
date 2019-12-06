import os

os.remove(".sconsign.dblite")

# Set our required libraries
libraries 		= []
library_paths 	= ''
cppDefines 		= {}
cppFlags 		= ['-Wall']#, '-Werror']
cxxFlags 		= ['-std=c++11']

# define the attributes of the build environment shared between
# both the debug and release builds
env = Environment()
env.Append(LIBS 			= libraries)
env.Append(LIBPATH 		= library_paths)
env.Append(CPPDEFINES 	= cppDefines)
env.Append(CPPFLAGS 		= cppFlags)
env.Append(CXXFLAGS 		= cxxFlags)

env.VariantDir('build/src','src',duplicate=0)

env.SConscript('src/SConscript', {'env': env})
