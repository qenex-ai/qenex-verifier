# Q-Lang Physics Example: 2D Ising Model
# Demonstrates phase transitions in magnetic systems

# Define physical constants
const BOLTZMANN_CONSTANT: Energy / Temperature = 1.380649e-23 J/K
const COUPLING_STRENGTH: Energy = 1.0 J

# Define lattice structure
type Lattice2D {
    size: Integer
    spins: Matrix<Integer>  # +1 or -1
    temperature: Temperature
}

# Define Hamiltonian for Ising model
law ising_hamiltonian {
    forall lattice: Lattice2D =>
        let H = -COUPLING_STRENGTH * sum_nearest_neighbors(lattice.spins)
        H is Energy
}

# Monte Carlo simulation
experiment ising_phase_transition {
    input:
        lattice_size: 100
        temperature_range: linspace(1.0, 5.0, 50) K
        mc_steps: 10000

    method: MonteCarlo {
        algorithm: "Metropolis-Hastings"
        observable: "magnetization"
    }

    output:
        magnetization_vs_temperature: Array<Float>
        critical_temperature: Temperature
        specific_heat: Array<Float>

    validate:
        # Verify critical temperature near theoretical value
        assert critical_temperature ≈ 2.269 K within 0.1 K
        # Check Curie-Weiss law above Tc
        assert magnetization follows_curie_weiss_law
}

# Conservation laws
law energy_conservation {
    forall system: Lattice2D =>
        derivative(total_energy(system), time) = 0
}

law detailed_balance {
    forall state_i, state_j: MicroState =>
        P(i→j) * exp(-E_i / (k_B * T)) = P(j→i) * exp(-E_j / (k_B * T))
}
