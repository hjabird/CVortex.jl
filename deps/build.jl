##############################################################################
#
# build.jl
#
# Build for cvortex.jl. Downloads correct precompiled binary.
#
# Copyright 2019-2020 HJA Bird
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to 
# deal in the Software without restriction, including without limitation the 
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in 
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
##############################################################################

using BinDeps
using Libdl

# version of cvortex package to use
cvortexver="0.3.7"

# binaries url prefix
url="https://github.com/hjabird/cvortex/releases/download/$cvortexver/"

tagfile = "installed_vers"
if !isfile(tagfile) || readchomp(tagfile) != "$cvortexver $WORD_SIZE"
    @info("Installing CVortex $cvortexver library.")
    working_dll = true
    opencl_dll = true
    if Sys.iswindows()
		@info("Downloading $url/cvortex_Win_x64_release.dll")
        run(download_cmd("$url/cvortex_Win_x64_release.dll", "libcvortex.dll"))
    elseif Sys.isapple()
        error("Sorry, no apple release of the binaries yet!")
        global working_dll = false
    elseif Sys.islinux()
		@info("Downloading $url/cvortex_Linux_x64_release.dll")
        run(download_cmd("$url/cvortex_Linux_x64_release.so", "libcvortex.so"))
    end

    try
        dlopen("libcvortex")
		@info("Successfully linked to library using dlopen.")
        open(tagfile, "w") do f
            println(f, "$cvortexver")
        end
    catch
		@info("Failed to successfully link to library using dlopen. Downloading "*
			"CVortex without OpenMP.")
        global opencl_dll = false    
        if Sys.iswindows()
		@info("Downloading $url/cvortex_Win_x64_release_NoOCL.dll")
            run(download_cmd("$url/cvortex_Win_x64_release_NoOCL.dll", "libcvortex.dll"))
        elseif Sys.islinux()
		@info("Downloading $url/cvortex_Linux_x64_release_NoOCL.so")
            run(download_cmd("$url/cvortex_Linux_x64_release_NoOCL.so", "libcvortex.so"))
        end
        try
            dlopen("libcvortex")
			@info("Successfully linked to library using dlopen. (No OpenCL version)")
            open(tagfile, "w") do f
                println(f, "$cvortexver NoOpenCL")
            end
        catch
			@info("Failed to successfully link to library using dlopen.")
            global working_dll = false
        end
    end
    
    if !working_dll
        error("Could not install working binary.")
    elseif !opencl_dll
        @warn("The OpenCL accelerated CVortex binary did not work. "*
            "An OpenMP only (no GPU acceleration) version was installed"*
            " instead. If you're expecting GPU acceleration to work, "*
            "check that OpenCL is properly installed on your platform.")
    end
else
    @info("CVortex $cvortexver is already installed.")
end
