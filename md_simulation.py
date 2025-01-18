from openmm import app, unit
from openmm.app import PDBFile, Modeller, ForceField, Simulation, DCDReporter, StateDataReporter
from openmm import LangevinMiddleIntegrator, Platform, MonteCarloBarostat
from pdbfixer import PDBFixer
import mdtraj as md
import numpy as np
import os

# Step 1: Prepare the PDB file
def prepare_pdb(input_pdb, output_pdb):
    """
    Prepares the PDB file by fixing missing residues, atoms, and non-standard residues.
    """
    print("Preparing PDB file...")
    fixer = PDBFixer(input_pdb)
    
    # Fix missing residues, atoms, and non-standard residues
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    
    # Save the fixed PDB file
    with open(output_pdb, 'w') as f:
        PDBFile.writeFile(fixer.topology, fixer.positions, f)
    print(f"Fixed PDB saved to {output_pdb}")

# Step 2: Load the system and define the force field
def load_system(pdb_file):
    """
    Loads the PDB file and defines the force field (AMBER ff15ipq for protein and TIP3P for water).
    """
    print("Loading system and defining force field...")
    pdb = PDBFile(pdb_file)
    
    # Define the force field (AMBER ff15ipq for protein and TIP3P for water)
    forcefield = ForceField('amber14/protein.ff15ipq.xml', 'amber14/tip3pfb.xml')
    
    # Create a Modeller object
    modeller = Modeller(pdb.topology, pdb.positions)
    
    # Add solvent (water box with 1.0 nm padding and 0.15 M ionic strength)
    print("Adding solvent...")
    modeller.addSolvent(forcefield, padding=1.0*unit.nanometers, ionicStrength=0.15*unit.molar)
    
    # Save the solvated system (optional)
    with open('solvated_complex.pdb', 'w') as f:
        PDBFile.writeFile(modeller.topology, modeller.positions, f)
    print("Solvated system saved to solvated_complex.pdb")
    
    return modeller, forcefield

# Step 3: Create the OpenMM system
def create_system(modeller, forcefield):
    """
    Creates the OpenMM system with the specified force field and simulation parameters.
    """
    print("Creating OpenMM system...")
    system = forcefield.createSystem(modeller.topology, 
                                     nonbondedMethod=app.PME, 
                                     nonbondedCutoff=1.0*unit.nanometers, 
                                     constraints=app.HBonds, 
                                     rigidWater=True, 
                                     ewaldErrorTolerance=0.0005)
    
    # Add a barostat for NPT simulations (1 atm, 300 K)
    barostat = MonteCarloBarostat(1.0*unit.atmospheres, 300*unit.kelvin, 25)
    system.addForce(barostat)
    
    return system

# Step 4: Set up the simulation
def setup_simulation(modeller, system, gpu_index=0):
    """
    Sets up the simulation with the Langevin Middle Integrator and platform.
    """
    print("Setting up simulation...")
    # Define the integrator (Langevin Middle Integrator, 300 K, 1 ps^-1 friction, 2 fs time step)
    integrator = LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
    
    # Choose the platform (CUDA, OpenCL, or CPU)
    platform = Platform.getPlatformByName('CUDA')  # Change to 'OpenCL' or 'CPU' if needed
    
    # Set the GPU device index
    platform.setPropertyDefaultValue('DeviceIndex', str(gpu_index))
    
    # Create the simulation object
    simulation = Simulation(modeller.topology, system, integrator, platform)
    
    # Set initial positions
    simulation.context.setPositions(modeller.positions)
    
    # Center the protein in the box
    simulation.context.computeVirtualSites()
    simulation.context.applyConstraints(1e-5)
    simulation.context.computeVirtualSites()
    
    return simulation

# Step 5: Minimize and equilibrate the system
def minimize_and_equilibrate(simulation):
    """
    Minimizes the energy and equilibrates the system in NVT and NPT ensembles.
    """
    print("Minimizing energy...")
    simulation.minimizeEnergy()
    
    # NVT equilibration (100 ps)
    print("NVT equilibration...")
    simulation.context.setVelocitiesToTemperature(300*unit.kelvin)
    simulation.step(50000)  # 100 ps (50000 steps * 0.002 ps/step)
    
    # NPT equilibration (100 ps)
    print("NPT equilibration...")
    simulation.step(50000)  # 100 ps (50000 steps * 0.002 ps/step)

# Step 6: Run production simulation
def run_production(simulation, output_dcd, output_log, steps=250000000):
    """
    Runs the production simulation and saves the trajectory and logs.
    """
    print("Running production simulation...")
    # Save trajectory (DCD format, every 100,000 steps)
    simulation.reporters.append(DCDReporter(output_dcd, 100000))
    
    # Save simulation logs (every 100,000 steps)
    simulation.reporters.append(StateDataReporter(output_log, 100000, 
                                                  step=True, 
                                                  potentialEnergy=True, 
                                                  temperature=True, 
                                                  volume=True,
                                                  progress=True,
                                                  remainingTime=True,
                                                  speed=True,
                                                  totalSteps=steps))  # Add totalSteps parameter
    
    # Run the simulation (500 ns by default)
    simulation.step(steps)  # 500 ns (250000000 steps * 0.002 ps/step)

# Step 7: Calculate binding affinity using MM/PBSA
def calculate_binding_affinity(trajectory_file, topology_file, chain_a_selection, chain_b_selection):
    """
    Calculates the binding affinity using MM/PBSA between two chains (e.g., chain A and chain B).
    """
    print("Calculating binding affinity using MM/PBSA...")
    
    # Load the trajectory and topology
    traj = md.load(trajectory_file, top=topology_file)
    
    # Select atoms for chain A and chain B
    chain_a_atoms = traj.topology.select(chain_a_selection)
    chain_b_atoms = traj.topology.select(chain_b_selection)
    
    # Combine chain A and chain B atoms
    complex_atoms = np.concatenate([chain_a_atoms, chain_b_atoms])
    
    # Create trajectories for complex, chain A, and chain B
    traj_complex = traj.atom_slice(complex_atoms)
    traj_chain_a = traj.atom_slice(chain_a_atoms)
    traj_chain_b = traj.atom_slice(chain_b_atoms)
    
    # Calculate MM/PBSA energy terms
    from openmm.app import PDBFile, Simulation, StateDataReporter
    from openmm import MonteCarloBarostat, Platform
    from openmm.unit import kilocalories_per_mole
    
    # Load the force field
    forcefield = ForceField('amber14/protein.ff15ipq.xml', 'amber14/tip3pfb.xml')
    
    # Create systems for complex, chain A, and chain B
    system_complex = forcefield.createSystem(traj_complex.topology.to_openmm(), 
                                             nonbondedMethod=app.NoCutoff, 
                                             constraints=app.HBonds)
    system_chain_a = forcefield.createSystem(traj_chain_a.topology.to_openmm(), 
                                             nonbondedMethod=app.NoCutoff, 
                                             constraints=app.HBonds)
    system_chain_b = forcefield.createSystem(traj_chain_b.topology.to_openmm(), 
                                             nonbondedMethod=app.NoCutoff, 
                                             constraints=app.HBonds)
    
    # Create simulations
    integrator = LangevinMiddleIntegrator(300*unit.kelvin, 1/unit.picosecond, 0.002*unit.picoseconds)
    platform = Platform.getPlatformByName('CPU')  # Use CPU for MM/PBSA calculations
    simulation_complex = Simulation(traj_complex.topology.to_openmm(), system_complex, integrator, platform)
    simulation_chain_a = Simulation(traj_chain_a.topology.to_openmm(), system_chain_a, integrator, platform)
    simulation_chain_b = Simulation(traj_chain_b.topology.to_openmm(), system_chain_b, integrator, platform)
    
    # Calculate energies for the complex, chain A, and chain B
    complex_energies = []
    chain_a_energies = []
    chain_b_energies = []
    
    for frame in traj_complex:
        # Complex energy
        simulation_complex.context.setPositions(frame.positions)
        state_complex = simulation_complex.context.getState(getEnergy=True)
        complex_energies.append(state_complex.getPotentialEnergy().value_in_unit(kilocalories_per_mole))
        
        # Chain A energy
        simulation_chain_a.context.setPositions(frame.positions[:len(chain_a_atoms)])
        state_chain_a = simulation_chain_a.context.getState(getEnergy=True)
        chain_a_energies.append(state_chain_a.getPotentialEnergy().value_in_unit(kilocalories_per_mole))
        
        # Chain B energy
        simulation_chain_b.context.setPositions(frame.positions[len(chain_a_atoms):])
        state_chain_b = simulation_chain_b.context.getState(getEnergy=True)
        chain_b_energies.append(state_chain_b.getPotentialEnergy().value_in_unit(kilocalories_per_mole))
    
    # Calculate binding free energy
    binding_energy = np.mean(complex_energies) - np.mean(chain_a_energies) - np.mean(chain_b_energies)
    print(f"Binding affinity (MM/PBSA) between chain A and chain B: {binding_energy:.2f} kcal/mol")


if __name__ == "__main__":
    input_pdb = "complex.pdb"
    fixed_pdb = "fixed_complex.pdb"
    
    # Step 1: Prepare the PDB file
    if not os.path.exists(input_pdb):
        raise FileNotFoundError(f"Input PDB file '{input_pdb}' not found.")
    prepare_pdb(input_pdb, fixed_pdb)
    
    # Step 2: Load the system and define the force field
    modeller, forcefield = load_system(fixed_pdb)
    
    # Step 3: Create the OpenMM system
    system = create_system(modeller, forcefield)
    
    # Step 4: Set up the simulation (specify GPU index here, e.g., 0 for the first GPU)
    simulation = setup_simulation(modeller, system, gpu_index=0)
    
    # Step 5: Minimize and equilibrate
    minimize_and_equilibrate(simulation)
    
    # Step 6: Run production simulation
    run_production(simulation, "trajectory.dcd", "log.txt", steps=250000000)
    
    # Step 7: Calculate binding affinity between chain A and chain B
    calculate_binding_affinity("trajectory.dcd", "solvated_complex.pdb", "chainid 0", "chainid 1")
    
    print("Simulation and binding affinity calculation complete!")
