# Q-Lang Biology Example: Protein Folding Simulation
# Demonstrates molecular dynamics for biological systems

# Import biological types
import qenex_bio.protein
import qenex_bio.forcefield

# Define amino acid sequence
type AminoAcidSequence {
    sequence: String  # Single-letter codes
    length: Integer
}

# Define protein structure
molecule Ubiquitin {
    sequence: "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"
    pdb_id: "1UBQ"
    num_residues: 76
    molecular_weight: 8565 Da
}

# Physical laws for protein folding
law hydrophobic_effect {
    forall residue: AminoAcid =>
        if residue.is_hydrophobic then
            residue.prefers_interior
        else
            residue.prefers_surface
}

law ramachandran_constraints {
    forall residue: AminoAcid =>
        # Phi and psi angles constrained by steric clashes
        (residue.phi, residue.psi) in allowed_regions
}

law anfinsen_principle {
    # Native structure is thermodynamic minimum
    forall protein: Protein =>
        protein.native_state = argmin(free_energy(protein))
}

# Molecular dynamics experiment
experiment ubiquitin_folding {
    input: Ubiquitin

    method: MolecularDynamics {
        forcefield: "AMBER99sb-ildn"
        water_model: "TIP3P"
        box_size: cubic(8.0 nm)
        periodic_boundary_conditions: true
    }

    protocol: {
        # Energy minimization
        minimize: {
            algorithm: "steepest_descent"
            max_steps: 50000
        }

        # NVT equilibration (constant volume, temperature)
        equilibrate_nvt: {
            temperature: 300 K
            duration: 100 ps
            coupling: "V-rescale"
        }

        # NPT equilibration (constant pressure, temperature)
        equilibrate_npt: {
            temperature: 300 K
            pressure: 1.0 bar
            duration: 100 ps
            coupling: "Parrinello-Rahman"
        }

        # Production run
        production: {
            temperature: 300 K
            pressure: 1.0 bar
            duration: 100 ns
            trajectory_output: every 10 ps
        }
    }

    output:
        trajectory: MDTrajectory
        rmsd_from_crystal: Array<Distance> in angstrom
        radius_of_gyration: Array<Distance> in angstrom
        secondary_structure: SecondaryStructureTimeline
        contact_map: Matrix<Boolean>

    validate:
        # RMSD should stabilize around experimental structure
        assert mean(rmsd_from_crystal[50ns:]) < 2.0 angstrom

        # Radius of gyration should be stable
        assert std_dev(radius_of_gyration[50ns:]) < 0.2 angstrom

        # Secondary structure should match experiment
        assert alpha_helix_content ≈ 26% within 5%
        assert beta_sheet_content ≈ 18% within 5%
}

# Conservation laws
law mass_conservation {
    forall system: MDSystem =>
        d(total_mass(system)) / dt = 0
}

law momentum_conservation {
    forall system: MDSystem =>
        if system.has_no_external_forces then
            d(total_momentum(system)) / dt = 0
}
