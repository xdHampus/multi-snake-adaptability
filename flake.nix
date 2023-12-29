{

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-23.05";
    utils.url = "github:numtide/flake-utils";
    utils.inputs.nixpkgs.follows = "nixpkgs";
  };

  outputs = { self, nixpkgs, ... }@inputs:
    inputs.utils.lib.eachSystem [ "x86_64-linux" ] (system:
      let
        pkgs = import nixpkgs {
          inherit system;
          overlays = [ ];
          config.allowUnfree = true;
        };
      in {
        devShell = pkgs.mkShell rec {

          packages = with pkgs; [
            python310
            python310Packages.pip
            python310Packages.virtualenv
            python310Packages.pyspark
            python310Packages.pandas
            python310Packages.numpy
            python310Packages.requests
            python310Packages.matplotlib
            python310Packages.pytorch
            python310Packages.tensorflow
            libstdcxx5
          ];
          shellHook = ''
            #alias venv="python3.10 -m venv .venv && source .venv/bin/activate"
            alias venv="virtualenv .venv && source .venv/bin/activate"
            alias pinstall="pip install -r requirements.txt"
            alias main="python3.10 src/main.py"
          '';
        };
      });
}