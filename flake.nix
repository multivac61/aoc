{
  description = "Using Nix Flake apps to run scripts with uv2nix";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    aoc-inputs = {
      url = "git+ssh://git@github.com/multivac61/aoc-inputs.git";
      flake = false;
    };

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };

    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    flake-parts.url = "github:hercules-ci/flake-parts";
    treefmt-nix.url = "github:numtide/treefmt-nix";
    treefmt-nix.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs =
    inputs@{
      flake-parts,
      aoc-inputs,
      nixpkgs,
      uv2nix,
      pyproject-nix,
      pyproject-build-systems,
      ...
    }:
    flake-parts.lib.mkFlake { inherit inputs; } {
      imports = [ inputs.treefmt-nix.flakeModule ];
      systems = [
        "x86_64-linux"
        "aarch64-darwin"
      ];
      perSystem =
        {
          config,
          self',
          inputs',
          pkgs,
          system,
          ...
        }:
        let
          inherit (nixpkgs) lib;
          inherit (lib) filterAttrs hasSuffix;

          workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

          overlay = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; };

          python = pkgs.python3;

          pythonSet =
            (pkgs.callPackage pyproject-nix.build.packages {
              inherit python;
            }).overrideScope
              (
                lib.composeManyExtensions [
                  pyproject-build-systems.overlays.default
                  overlay
                ]
              );

          venv = pythonSet.mkVirtualEnv "aoc-default-env" workspace.deps.default;
        in
        {
          treefmt = {
            projectRootFile = "flake.nix";
            programs.ruff-check.enable = true;
            programs.ruff-format.enable = true;
            programs.nixfmt.enable = true;
            programs.mdformat.enable = true;
            programs.toml-sort.enable = true;
          };
          packages =
            let
              basedir = ./year;

              # Get a list of regular Python files in directory
              files = filterAttrs (name: type: type == "regular" && hasSuffix ".py" name) (
                builtins.readDir basedir
              );
              solutions =
                # Map over files to:
                # - Rewrite script shebangs as shebangs pointing to the virtualenv
                # - Strip .py suffixes from attribute names
                #   Making a script "greet.py" runnable as "nix run .#greet"
                lib.mapAttrs' (
                  name: _:
                  lib.nameValuePair (lib.removeSuffix ".py" name) (
                    let
                      script = basedir + "/${name}";
                      baseProgram = pkgs.runCommand name { buildInputs = [ venv ]; } ''
                        cp ${script} $out
                        chmod +x $out
                        patchShebangs $out
                      '';
                    in
                    pkgs.writeShellApplication {
                      inherit name;
                      runtimeInputs = [ baseProgram ];
                      text = ''
                        if [ ! -d inputs ]; then
                          ln -s ${aoc-inputs}/inputs inputs
                        fi
                        ${baseProgram}
                      '';
                    }
                  )
                ) files;
            in
            solutions;
          # This example provides two different modes of development:
          # - Impurely using uv to manage virtual environments
          # - Pure development using uv2nix to manage virtual environments
          devShells = {
            # It is of course perfectly OK to keep using an impure virtualenv workflow and only use uv2nix to build packages.
            # This devShell simply adds Python and undoes the dependency leakage done by Nixpkgs Python infrastructure.
            impure = pkgs.mkShell {
              packages = [
                python
                pkgs.uv
              ];
              env =
                {
                  # Prevent uv from managing Python downloads
                  UV_PYTHON_DOWNLOADS = "never";
                  # Force uv to use nixpkgs Python interpreter
                  UV_PYTHON = python.interpreter;
                }
                // lib.optionalAttrs pkgs.stdenv.isLinux {
                  # Python libraries often load native shared objects using dlopen(3).
                  # Setting LD_LIBRARY_PATH makes the dynamic library loader aware of libraries without using RPATH for lookup.
                  LD_LIBRARY_PATH = lib.makeLibraryPath pkgs.pythonManylinuxPackages.manylinux1;
                };
              shellHook = ''
                unset PYTHONPATH
                # Link inputs from flake input if not exists
                if [ ! -d inputs ]; then
                  ln -s ${aoc-inputs}/inputs inputs
                fi
              '';
            };

            # This devShell uses uv2nix to construct a virtual environment purely from Nix, using the same dependency specification as the application.
            # The notable difference is that we also apply another overlay here enabling editable mode ( https://setuptools.pypa.io/en/latest/userguide/development_mode.html ).
            #
            # This means that any changes done to your local files do not require a rebuild.
            uv2nix =
              let
                # Create an overlay enabling editable mode for all local dependencies.
                editableOverlay = workspace.mkEditablePyprojectOverlay {
                  # Use environment variable
                  root = "$REPO_ROOT";
                  # Optional: Only enable editable for these packages
                  # members = [ "hello-world" ];
                };

                # Override previous set with our overrideable overlay.
                editablePythonSet = pythonSet.overrideScope editableOverlay;

                # Build virtual environment, with local packages being editable.
                #
                # Enable all optional dependencies for development.
                virtualenv = editablePythonSet.mkVirtualEnv "hello-world-dev-env" workspace.deps.all;

              in
              pkgs.mkShell {
                packages = [
                  virtualenv
                  pkgs.uv
                ];

                env = {
                  # Don't create venv using uv
                  UV_NO_SYNC = "1";

                  # Force uv to use Python interpreter from venv
                  UV_PYTHON = "${virtualenv}/bin/python";

                  # Prevent uv from downloading managed Python's
                  UV_PYTHON_DOWNLOADS = "never";
                };

                shellHook = ''
                  if [ ! -d inputs ]; then
                    ln -s ${aoc-inputs}/inputs inputs
                  fi

                  # Undo dependency propagation by nixpkgs.
                  unset PYTHONPATH

                  # Get repository root using git. This is expanded at runtime by the editable `.pth` machinery.
                  export REPO_ROOT=$(git rev-parse --show-toplevel)
                '';
              };
          };
        };
    };
}
