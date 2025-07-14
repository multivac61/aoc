{
  description = "Advent of Code solutions using uv2nix";

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

    treefmt-nix.url = "github:numtide/treefmt-nix";
  };

  outputs =
    {
      self,
      nixpkgs,
      uv2nix,
      pyproject-nix,
      pyproject-build-systems,
      aoc-inputs,
      treefmt-nix,
      ...
    }:
    let
      inherit (nixpkgs) lib;

      supportedSystems = [
        "x86_64-linux"
        "aarch64-darwin"
      ];

      forAllSystems = f: lib.genAttrs supportedSystems f;
      treefmtEval = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
        in
        treefmt-nix.lib.evalModule pkgs {
          programs.ruff-check.enable = true;
          programs.ruff-format.enable = true;
          programs.deadnix.enable = true;
          programs.nixfmt.enable = true;
          programs.statix.enable = true;
        }
      );
    in
    {
      packages = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};

          # Load uv workspace
          workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

          # Create package overlay from workspace
          overlay = workspace.mkPyprojectOverlay {
            sourcePreference = "wheel";
          };

          # Use Python 3.12 from nixpkgs
          python = pkgs.python312;

          # Construct package set
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

          # Create virtual environment
          venv = pythonSet.mkVirtualEnv "aoc-env" workspace.deps.default;

          # Get year files
          basedir = ./year;
          files = lib.filterAttrs (name: type: type == "regular" && lib.hasSuffix ".py" name) (
            builtins.readDir basedir
          );

          # Create packages for each year
          yearPackages = lib.mapAttrs' (
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
        yearPackages
        // {
          default = venv;
        }
      );

      checks = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};

          # Load uv workspace
          workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };

          # Create package overlay from workspace
          overlay = workspace.mkPyprojectOverlay {
            sourcePreference = "wheel";
          };

          # Use Python 3.12 from nixpkgs
          python = pkgs.python312;

          # Construct package set
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

          # Create virtual environment
          venv = pythonSet.mkVirtualEnv "aoc-env" workspace.deps.default;

          # Get year files
          basedir = ./year;
          files = lib.filterAttrs (name: type: type == "regular" && lib.hasSuffix ".py" name) (
            builtins.readDir basedir
          );

          # Create checks for each year
          yearChecks = lib.mapAttrs' (
            name: _:
            lib.nameValuePair (lib.removeSuffix ".py" name) (
              pkgs.runCommand "check-${lib.removeSuffix ".py" name}"
                {
                  buildInputs = [ venv ];
                }
                ''
                  # Create a working directory with the inputs
                  mkdir -p work
                  cd work
                  cp -r ${./.}/* .
                  chmod -R +w .
                  ln -s ${aoc-inputs}/inputs inputs
                  export PYTHONPATH="${venv}/${python.sitePackages}:$PYTHONPATH"
                  ${python.interpreter} ${basedir}/${name}
                  echo "SUCCESS: ${name} completed without errors" > $out
                ''
            )
          ) files;
        in
        yearChecks // { formatting = treefmtEval.${system}.config.build.check self; }
      );

      formatter = forAllSystems (system: treefmtEval.${system}.config.build.wrapper);

      devShells = forAllSystems (
        system:
        let
          pkgs = nixpkgs.legacyPackages.${system};
          python = pkgs.python312;
        in
        {
          default = pkgs.mkShell {
            packages = [
              python
              pkgs.uv
            ];
            env = {
              UV_PYTHON_DOWNLOADS = "never";
              UV_PYTHON = python.interpreter;
            };
            shellHook = ''
              unset PYTHONPATH
              if [ ! -d inputs ]; then
                ln -s ${aoc-inputs}/inputs inputs
              fi
            '';
          };
        }
      );
    };
}
