"""
Environment-Aware Dynamic Weighting (EADW) Module
===================================================
Part of the R-MMAF (Robust Multi-Modal Affective-Aware Framework)

Dynamically fuses RGB visual and Thermal IR sensor feeds based on
real-time environmental conditions (fog, precipitation, illumination).

Author: Sanchit Agarwal
"""


class EnvironmentSimulator:
    """
    Simulates real-world environmental conditions that affect sensor performance.

    Parameters
    ----------
    fog_density : float
        Fog intensity from 0.0 (clear) to 1.0 (dense fog)
    precipitation : float
        Rain/snow intensity from 0.0 (dry) to 1.0 (heavy storm)
    illumination : float
        Lighting level from 1.0 (bright daylight) to 0.0 (pitch black)
    """

    def __init__(self, fog_density=0.0, precipitation=0.0, illumination=1.0):
        self.fog_density = fog_density
        self.precipitation = precipitation
        self.illumination = illumination


def calculate_eadw_weights(env: EnvironmentSimulator):
    """
    Calculates the dynamic trust weights for each sensor channel.

    The degradation score is the worst-case reading across all
    environmental factors. Thermal IR weight increases as visual
    conditions deteriorate — ensuring the drone is never lost in
    adverse weather.

    Parameters
    ----------
    env : EnvironmentSimulator
        Current environmental conditions

    Returns
    -------
    tuple : (visual_weight, thermal_weight)
        Normalized weights that always sum to 1.0
    """
    # Max degradation across all environmental factors
    degradation_score = max(
        env.fog_density,
        env.precipitation,
        1.0 - env.illumination   # Less light = more degradation
    )

    thermal_weight = degradation_score
    visual_weight = 1.0 - thermal_weight

    return round(visual_weight, 2), round(thermal_weight, 2)


# ----- Quick Demo -----
if __name__ == "__main__":
    scenarios = {
        "Sunny Day":      EnvironmentSimulator(fog_density=0.0, precipitation=0.0, illumination=1.0),
        "Midnight Fog":   EnvironmentSimulator(fog_density=0.9, precipitation=0.0, illumination=0.1),
        "Heavy Storm":    EnvironmentSimulator(fog_density=0.4, precipitation=0.9, illumination=0.5),
        "Overcast Night": EnvironmentSimulator(fog_density=0.2, precipitation=0.1, illumination=0.05),
    }

    print(f"{'Scenario':<20} {'Visual Weight':>15} {'Thermal Weight':>16}")
    print("-" * 53)
    for name, env in scenarios.items():
        v, t = calculate_eadw_weights(env)
        print(f"{name:<20} {v:>15.2f} {t:>16.2f}")
