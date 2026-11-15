def ordered_measurements(packet):
    return [(release["release_id"], row) for release in packet["releases"] for row in release["measurements"]]


def noise_covariance(packet):
    measurements = ordered_measurements(packet)
    covariance = []
    for first_id, first in measurements:
        covariance.append([sum(coefficient * second["loading"].get(noise, 0) * packet["noise_variance"][noise]
                               for noise, coefficient in first["loading"].items()) for second_id, second in measurements])
    return covariance


def release_profile(packet):
    measurements = ordered_measurements(packet)
    covariance = noise_covariance(packet)
    release_ids = {row["release_id"] for row in packet["releases"]}
    replaced = {row["replaces"] for row in packet["releases"] if row["replaces"] is not None}
    missing_parents = sorted(replaced - release_ids)
    cross_release_pairs = []
    for first_index, (first_id, first) in enumerate(measurements):
        for second_index in range(first_index + 1, len(measurements)):
            second_id, second = measurements[second_index]
            if first_id != second_id and covariance[first_index][second_index]:
                cross_release_pairs.append([first["measurement"], second["measurement"]])
    return {"measurement_order": [row["measurement"] for release_id, row in measurements],
            "known_noise_covariance": covariance, "correlated_cross_release_pairs": cross_release_pairs,
            "current_releases": sorted(release_ids - replaced), "missing_revision_parents": missing_parents,
            "latent_cell_order": [row["cell_id"] for row in packet["cells"]],
            "measurement_design": [[row["coefficients"].get(cell["cell_id"], 0) for cell in packet["cells"]]
                                   for release_id, row in measurements]}
