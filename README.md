
# Molecular Dynamics Simulation with OpenMM

This repository contains a Python script for running molecular dynamics (MD) simulations using OpenMM. The code prepares a protein system, runs MD simulations, and calculates binding affinity between two chains (e.g., protein-peptide or protein-protein interactions) using MM/PBSA.

## Features
- **PDB Preparation**: Fixes missing residues, atoms, and non-standard residues using `PDBFixer`.
- **System Setup**: Solvates the system, adds ions, and defines the force field (AMBER ff15ipq for protein and TIP3P for water).
- **Simulation**: Runs energy minimization, equilibration (NVT and NPT), and production MD simulations.
- **GPU Support**: Allows specifying which GPU to use for the simulation.
- **Binding Affinity Calculation**: Computes binding affinity between two chains (e.g., chain A and chain B) using MM/PBSA.

## Requirements
- Python 3.7+
- OpenMM
- PDBFixer
- MDTraj
- NumPy

Install dependencies using:
```bash
pip install openmm pdbfixer mdtraj numpy
```

## Usage
1. Place your input PDB file (e.g., `complex.pdb`) in the working directory.
2. Run the script:
   ```bash
   python md_simulation.py
   ```
3. The script will:
   - Prepare the PDB file.
   - Solvate the system and add ions.
   - Run energy minimization and equilibration.
   - Perform a production MD simulation.
   - Calculate binding affinity between two specified chains.

## Customization
- **Input PDB**: Replace `complex.pdb` with your input PDB file.
- **GPU Selection**: Specify the GPU index in the `setup_simulation` function (default is `0`).
- **Chain Selection**: Modify the chain selection strings in the `calculate_binding_affinity` function (e.g., `"chainid 0"` for chain A and `"chainid 1"` for chain B).

## Output Files
- `fixed_complex.pdb`: Prepared and solvated PDB file.
- `trajectory.dcd`: Trajectory file from the production simulation.
- `log.txt`: Simulation logs (energy, temperature, volume, etc.).
- Binding affinity results printed to the console.

## Example
To calculate binding affinity between chain A and chain B:
```python
calculate_binding_affinity("trajectory.dcd", "solvated_complex.pdb", "chainid 0", "chainid 1")
```

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
