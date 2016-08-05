using BinDeps
using Compat

@BinDeps.setup

windllname = "libminotaur-1"
libminotaur = library_dependency("libminotaur", aliases=[windllname])

minotaurname = "minotaur-0.2.0-src"

provides(Sources, URI("http://www.mcs.anl.gov/research/projects/minotaur/dist/minotaur-0.2.0-src.tar.gz"),
    libminotaur, os = :Unix)

prefix   = joinpath(BinDeps.depsdir(libminotaur),"usr")
patchdir = BinDeps.depsdir(libminotaur)
srcdir   = joinpath(BinDeps.depsdir(libminotaur),"src",minotaurname)

provides(SimpleBuild,
    (@build_steps begin
        GetSources(libminotaur)
        @build_steps begin
            @build_steps begin
                ChangeDirectory(patchdir)
                `./get.Thirdparty`
            end
            ChangeDirectory(srcdir)
            `cmake -DALL_EXTERNS_PATH:PATH=$patchdir/third-party .`
            `make install`
        end
    end),
libminotaur, os = :Unix)

#=
# OS X
@osx_only begin
    using Homebrew
    provides(Homebrew.HB, "staticfloat/juliadeps/minotaur", libminotaur, os = :Darwin)
end


# Windows
@windows_only begin
    using WinRPM
    provides(WinRPM.RPM, "Minotaur", [libminotaur], os = :Windows)
end
=#

@BinDeps.install Dict(:libminotaur => :libminotaur)