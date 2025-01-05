{
  description = "Using Nix Flake apps to run scripts with uv2nix";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

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
            programs.shellcheck.enable = true;
            programs.mdformat.enable = true;
            programs.toml-sort.enable = true;
          };
          packages = {
            download-all = pkgs.writeShellApplication {
              name = "download-all";

              runtimeInputs = with pkgs; [
                curl
                parallel
              ];

              text = ''
                if [ -z "''${AOC_SESSION:-}" ]; then
                  echo "Error: AOC_SESSION must be set"
                  exit 1
                fi

                mkdir -p inputs

                YEARS=$(seq 2015 2024)

                download_year() {
                  local year=$1
                  echo "Downloading year $year..."
                  mkdir -p "inputs/$year"
                  
                  # Only download days that don't exist yet
                  seq 1 25 | parallel -j 8 \
                    "if [ ! -f inputs/$year/{} ]; then \
                      curl -s 'https://adventofcode.com/$year/day/{}/input' \
                      -H 'Cookie: session=$AOC_SESSION' \
                      -o 'inputs/$year/{}' || \
                      echo 'Failed to download year $year day {}'; \
                    fi"
                }

                export -f download_year
                echo "$YEARS" | parallel -j 4 download_year
              '';
            };
            download-year = pkgs.writeShellApplication {
              name = "download-year";

              runtimeInputs = with pkgs; [
                curl
                parallel
              ];

              text = ''
                if [ -z "''${YEAR:-}" ]; then
                  echo "Error: YEAR must be set"
                  exit 1
                fi

                mkdir -p "''${YEAR}"
                seq 1 25 | parallel -j 8 \
                  "curl 'https://adventofcode.com/''${YEAR}/day/{}/input' \
                  -H 'Cookie: session=''${AOC_SESSION}' \
                  -o 'inputs/''${YEAR}/{}'"
              '';
            };
            download-day = pkgs.writeShellApplication {
              name = "download-day";

              runtimeInputs = with pkgs; [
                curl
              ];

              text = ''
                if [ -z "''${YEAR:-}" ]; then
                  echo "Error: YEAR must be set"
                  exit 1
                fi

                if [ -z "''${DAY:-}" ]; then
                  echo "Error: DAY must be set"
                  exit 1
                fi

                curl "https://adventofcode.com/''${YEAR}/day/''${DAY}/input" \
                  -H "Cookie: session=''${AOC_SESSION}" \
                  -o "inputs/''${YEAR}/''${DAY}" \
                  --create-dirs
              '';
            };
          };
          apps =
            let
              basedir = ./year;

              # Get a list of regular Python files in directory
              files = filterAttrs (name: type: type == "regular" && hasSuffix ".py" name) (
                builtins.readDir basedir
              );

            in
            # Map over files to:
            # - Rewrite script shebangs as shebangs pointing to the virtualenv
            # - Strip .py suffixes from attribute names
            #   Making a script "greet.py" runnable as "nix run .#greet"
            lib.mapAttrs' (
              name: _:
              lib.nameValuePair (lib.removeSuffix ".py" name) (
                let
                  script = basedir + "/${name}";

                  # Patch script shebang
                  program = pkgs.runCommand name { buildInputs = [ venv ]; } ''
                    cp ${script} $out
                    chmod +x $out
                    patchShebangs $out
                  '';
                in
                {
                  type = "app";
                  program = "${program}";
                  meta = {
                    description = "Solution for ${lib.removeSuffix ".py" name}";
                    license = lib.licenses.mit;
                  };
                }
              )
            ) files;
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
