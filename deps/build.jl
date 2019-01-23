using BinDeps
using Compat

# version of cubature package to use
cvortexver="0.1.0"

# binaries url prefix
url="https://github.com/hjabird/cvortex/releases/download/v$cvortexver/"

tagfile = "installed_vers"
if !isfile(tagfile) || readchomp(tagfile) != "$cvortexver $WORD_SIZE"
    @info("Installing cvortex $cvortexver library...")
    if Sys.iswindows()
        run(download_cmd("$url/cvortex.dll", "libcvortex.dll"))
    elseif Sys.isapple()
        info("Sorry, no apple release of the binaries yet!")
    elseif Sys.islinux()
        run(download_cmd("$url/libcvortex.so", "libcvortex.so"))
    end
    open(tagfile, "w") do f
        println(f, "$cvortexver")
    end
else
    @info("cvortex $cvortexver is already installed.")
end