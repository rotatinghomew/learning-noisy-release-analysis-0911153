from app.latent import latent_participation
from app.releases import release_profile
from app.protocol import finite, unique


def validate(packet):
    if not packet["revision"]:
        raise ValueError("missing revision")
    unique(packet["cells"], "cell_id")
    unique(packet["releases"], "release_id")
    cells = {row["cell_id"] for row in packet["cells"]}
    for release in packet["releases"]:
        for row in release["measurements"]:
            finite(row["value"], "released value")
            if set(row["coefficients"]) - cells or set(row["loading"]) - set(packet["noise_variance"]):
                raise ValueError("unknown count cell or noise variable")


def release_inventory(packet, options):
    releases = []
    for release in packet["releases"]:
        variances = [{"measurement": row["measurement"], "released_value": row["value"],
                      "marginal_noise_variance": sum(coefficient ** 2 * packet["noise_variance"][noise]
                                                     for noise, coefficient in row["loading"].items())}
                     for row in release["measurements"]]
        releases.append({"release_id": release["release_id"], "replaces": release["replaces"],
                         "measurements": variances})
    return {"source_profile": release_profile(packet), "releases": releases, "interpretation": "Values are noisy releases, not latent participant counts."}


ANALYSES = {"release_inventory": release_inventory,
            "latent_participation": latent_participation}
