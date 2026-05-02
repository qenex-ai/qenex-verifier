# Q-Lang Astronomy Example: Three-Body Problem
# Demonstrates celestial mechanics and orbital dynamics

import qenex_astro.celestial_mechanics
import qenex_astro.coordinates

# Physical constants
const G: Float = 6.67430e-11 m^3/(kg*s^2)  # Gravitational constant
const AU: Distance = 1.495978707e11 m       # Astronomical unit
const M_SUN: Mass = 1.98847e30 kg          # Solar mass
const M_EARTH: Mass = 5.97217e24 kg        # Earth mass
const M_MOON: Mass = 7.342e22 kg           # Lunar mass

# Define celestial body type
type CelestialBody {
    name: String
    mass: Mass
    position: Vector3<Distance>
    velocity: Vector3<Velocity>
    radius: Distance
}

# Newton's law of universal gravitation
law gravitational_force {
    forall body1, body2: CelestialBody =>
        let r = body2.position - body1.position
        let F = -G * body1.mass * body2.mass / (|r|^2) * normalize(r)
        F is Force
}

# Kepler's laws
law keplers_first_law {
    forall orbit: Orbit =>
        orbit.shape = ellipse or orbit.shape = parabola or orbit.shape = hyperbola
}

law keplers_second_law {
    # Equal areas in equal times
    forall orbit: Orbit =>
        d(orbital_area(orbit)) / dt = constant
}

law keplers_third_law {
    forall orbit: Orbit =>
        orbit.period^2 / orbit.semi_major_axis^3 = 4π^2 / (G * M_central)
}

# Three-body simulation: Sun-Earth-Moon
experiment three_body_dynamics {
    input: {
        # Initial conditions (J2000.0 epoch)
        sun: CelestialBody {
            name: "Sun"
            mass: M_SUN
            position: (0, 0, 0) m
            velocity: (0, 0, 0) m/s
            radius: 6.96e8 m
        }

        earth: CelestialBody {
            name: "Earth"
            mass: M_EARTH
            position: (1.0 * AU, 0, 0) m
            velocity: (0, 29.78e3, 0) m/s
            radius: 6.371e6 m
        }

        moon: CelestialBody {
            name: "Moon"
            mass: M_MOON
            position: (1.0 * AU + 3.844e8, 0, 0) m
            velocity: (0, 29.78e3 + 1.022e3, 0) m/s
            radius: 1.737e6 m
        }
    }

    method: NBodySimulation {
        integrator: "Runge-Kutta-Fehlberg-7(8)"  # RKF78
        adaptive_timestep: true
        timestep_initial: 60 s
        timestep_min: 1 s
        timestep_max: 3600 s
        tolerance: 1e-12
    }

    simulate: {
        duration: 365.25 days
        output_interval: 1 hour
    }

    output:
        trajectories: Map<String, Array<StateVector>>
        orbital_elements: Map<String, OrbitalElements>
        total_energy: Array<Energy>
        angular_momentum: Array<Vector3<AngularMomentum>>

    validate:
        # Energy conservation (within numerical error)
        let E_initial = total_energy[0]
        let E_final = total_energy[-1]
        assert |E_final - E_initial| / |E_initial| < 1e-8

        # Angular momentum conservation
        let L_initial = angular_momentum[0]
        let L_final = angular_momentum[-1]
        assert |L_final - L_initial| / |L_initial| < 1e-8

        # Earth completes approximately one orbit
        let earth_period = estimate_period(trajectories["Earth"])
        assert earth_period ≈ 365.25 days within 0.1 days

        # Moon completes approximately 12.4 orbits around Earth
        let moon_period = estimate_period_around_earth(trajectories["Moon"])
        assert moon_period ≈ 27.3 days within 0.5 days
}

# Orbital elements calculation
experiment calculate_orbital_elements {
    input: trajectories from three_body_dynamics

    compute:
        forall body in ["Earth", "Moon"]:
            elements[body] = {
                semi_major_axis: calculate_sma(trajectories[body])
                eccentricity: calculate_eccentricity(trajectories[body])
                inclination: calculate_inclination(trajectories[body])
                longitude_of_ascending_node: calculate_lan(trajectories[body])
                argument_of_periapsis: calculate_aop(trajectories[body])
                true_anomaly: calculate_ta(trajectories[body])
            }

    validate:
        # Earth's orbit should be nearly circular
        assert elements["Earth"].eccentricity ≈ 0.0167 within 0.001

        # Earth's semi-major axis should be 1 AU
        assert elements["Earth"].semi_major_axis ≈ AU within 0.01 * AU
}

# Conservation laws
law energy_conservation {
    forall system: NBodySystem =>
        d(kinetic_energy(system) + potential_energy(system)) / dt = 0
}

law angular_momentum_conservation {
    forall system: NBodySystem with no_external_torques =>
        d(total_angular_momentum(system)) / dt = 0
}

law center_of_mass_conservation {
    forall system: NBodySystem =>
        total_mass * center_of_mass_velocity = constant
}
