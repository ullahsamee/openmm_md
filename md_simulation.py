#!/usr/bin/env python
from openmm import app, unit
from openmm.app import PDBFile, Modeller, ForceField, Simulation, DCDReporter, StateDataReporter
from openmm import LangevinMiddleIntegrator, Platform, MonteCarloBarostat
from pdbfixer import PDBFixer
import mdtraj as md
import numpy as np
import os
import argparse
from sys import exit

def parse_arguments():
    """Parse command-line arguments for simulation configuration"""
    parser = argparse.ArgumentParser(description="Molecular Dynamics Simulation Workflow")
    
    parser.add_argument("-i", "--input", required=True, help="Input PDB file")
    parser.add_argument("-o", "--output", default="output", help="Base name for output files")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    parser.add_argument("--steps", type=int, default=250000000, 
                      help="Production simulation steps (default: 500ns)")
    parser.add_argument("--chains", nargs=2, default=["chainid 0", "chainid 1"],
                      help="Chain selections for MM/PBSA calculation")
    
    return parser.parse_args()

def prepare_pdb(input_pdb, output_pdb):
    """Fix PDB structure and add missing components"""
    if not os.path.exists(input_pdb):
        raise FileNotFoundError(f"Input PDB {input_pdb} not found")
    
    print(f"\n{' PREPARING PDB STRUCTURE ':=^80}")
    fixer = PDBFixer(input_pdb)
    
    # Apply standard fixes
    operations = [
        ("Finding missing residues", fixer.findMissingResidues),
        ("Identifying non-standard residues", fixer.findNonstandardResidues),
        ("Replacing non-standard residues", fixer.replaceNonstandardResidues),
        ("Locating missing atoms", fixer.findMissingAtoms),
        ("Adding missing atoms", fixer.addMissingAtoms)
    ]
    
    for desc, op in operations:
        print(f"> {desc}")
        op()
    
    print(f"\nSaving fixed PDB to {output_pdb}")
    with open(output_pdb, 'w') as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)

def setup_system(pdb_file, output_base):
    """Initialize molecular system and force field"""
    print(f"\n{' SYSTEM SETUP ':=^80}")
    pdb = PDBFile(pdb_file)
    forcefield = ForceField('amber14/protein.ff15ipq.xml', 'amber14/tip3pfb.xml')
    
    print("Creating molecular model...")
    modeller = Modeller(pdb.topology, pdb.positions)
    
    print("\nSolvating system with TIP3P water:")
    print(f"- Padding: 1.0 nm\n- Ionic strength: 0.15 M")
    modeller.addSolvent(forcefield, padding=1.0*unit.nanometers, 
                      ionicStrength=0.15*unit.molar)
    
    solvated_pdb = f"{output_base}_solvated.pdb"
    print(f"Saving solvated system to {solvated_pdb}")
    with open(solvated_pdb, 'w') as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f)
    
    return modeller, forcefield, solvated_pdb

def configure_simulation(modeller, forcefield, gpu_index=0):
    """Create and configure OpenMM simulation"""
    print(f"\n{' SIMULATION CONFIGURATION ':=^80}")
    print("Creating system with force field parameters:")
    print("- PME electrostatics\n- 1.0 nm cutoff\n- HBond constraints\n- Rigid water")
    system = forcefield.createSystem(modeller.topology,
                                   nonbondedMethod=app.PME,
                                   nonbondedCutoff=1.0*unit.nanometers,
                                   constraints=app.HBonds,
                                   rigidWater=True,
                                   ewaldErrorTolerance=0.0005)
    
    print("\nAdding barostat for NPT ensemble (1 atm, 300 K)")
    system.addForce(MonteCarloBarostat(1.0*unit.atmospheres, 300*unit.kelvin, 25))
    
    print("\nConfiguring integration parameters:")
    print("- Langevin Middle Integrator\n- 300 K\n- 1 ps⁻¹ friction\n- 2 fs timestep")
    integrator = LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 
                                        0.002*unit.picoseconds)
    
    platform = Platform.getPlatformByName('CUDA')
    platform.setPropertyDefaultValue('DeviceIndex', str(gpu_index))
    
    simulation = Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    
    print("\nSystem initialization complete")
    return simulation

def run_simulation(simulation, output_base, production_steps):
    """Execute energy minimization and MD simulation"""
    print(f"\n{' RUNNING SIMULATION ':=^80}")
    
    # Energy minimization
    print("\nStarting energy minimization")
    simulation.minimizeEnergy()
    
    # Equilibration phases
    print("\nEquilibration phases:")
    for phase, steps in [("NVT", 50000), ("NPT", 50000)]:
        print(f"- {phase} equilibration (100 ps)")
        simulation.context.setVelocitiesToTemperature(300*unit.kelvin)
        simulation.step(steps)
    
    # Production run
    traj_file = f"{output_base}.dcd"
    log_file = f"{output_base}.log"
    
    print(f"\nStarting production run ({production_steps} steps)")
    simulation.reporters = [
        DCDReporter(traj_file, 100000),
        StateDataReporter(log_file, 100000,
                        step=True,
                        potentialEnergy=True,
                        temperature=True,
                        volume=True,
                        progress=True,
                        remainingTime=True,
                        speed=True,
                        totalSteps=production_steps)
    ]
    
    simulation.step(production_steps)
    return traj_file, log_file

def calculate_binding_affinity(traj_file, top_file, chain_selections):
    """Perform MM/PBSA binding free energy calculation"""
    print(f"\n{' BINDING AFFINITY CALCULATION ':=^80}")
    
    print("Loading trajectory and processing...")
    traj = md.load(traj_file, top=top_file)
    
    print("\nSelecting protein chains:")
    selections = [f"Chain selection {i+1}: {sel}" 
                for i, sel in enumerate(chain_selections)]
    print("\n".join(selections))
    
    chain_a = traj.topology.select(chain_selections[0])
    chain_b = traj.topology.select(chain_selections[1])
    complex_atoms = np.concatenate([chain_a, chain_b])
    
    print("\nCalculating potential energies:")
    energy_components = {
        "Complex": traj.atom_slice(complex_atoms),
        "Chain A": traj.atom_slice(chain_a),
        "Chain B": traj.atom_slice(chain_b)
    }
    
    results = {}
    for name, system in energy_components.items():
        print(f"- Processing {name}")
        ff = ForceField('amber14/protein.ff15ipq.xml', 'amber14/tip3pfb.xml')
        omm_top = system.topology.to_openmm()
        
        system = ff.createSystem(omm_top, 
                               nonbondedMethod=app.NoCutoff,
                               constraints=app.HBonds)
        
        integrator = LangevinMiddleIntegrator(300*unit.kelvin, 
                                           1/unit.picosecond, 
                                           0.002*unit.picoseconds)
        simulation = Simulation(omm_top, system, integrator, 
                              Platform.getPlatformByName('CPU'))
        
        energies = []
        for frame in system.positions:
            simulation.context.setPositions(frame)
            state = simulation.context.getState(getEnergy=True)
            energies.append(state.getPotentialEnergy().value_in_unit(
                unit.kilocalories_per_mole))
        
        results[name] = np.mean(energies)
    
    binding_energy = results["Complex"] - results["Chain A"] - results["Chain B"]
    print(f"\nFinal binding affinity: {binding_energy:.2f} kcal/mol")
    return binding_energy

def main():
    args = parse_arguments()
    
    try:
        # Structure preparation
        fixed_pdb = f"{args.output}_fixed.pdb"
        prepare_pdb(args.input, fixed_pdb)
        
        # System setup
        modeller, forcefield, solvated_pdb = setup_system(fixed_pdb, args.output)
        
        # Simulation configuration
        simulation = configure_simulation(modeller, forcefield, args.gpu)
        
        # Run simulation
        traj_file, log_file = run_simulation(simulation, args.output, args.steps)
        
        # Binding affinity calculation
        calculate_binding_affinity(traj_file, solvated_pdb, args.chains)
        
        print("\nSimulation workflow completed successfully")
        
    except Exception as e:
        print(f"\nERROR: {str(e)}")
        exit(1)

if __name__ == "__main__":
    main()
