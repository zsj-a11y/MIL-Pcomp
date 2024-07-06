import numpy as np
import math


def build_data(positive_data, negative_data, prior, n):
    positive_data = np.array(positive_data)
    negative_data = np.array(negative_data)


    pp_prior = prior * prior
    pn_prior = prior * (1 - prior)
    nn_prior = (1 - prior) * (1 - prior)
    total_prior = pn_prior + pp_prior + nn_prior
    pn_number = math.floor(n * (pn_prior / total_prior))
    pp_number = math.floor(n * (pp_prior / total_prior))
    nn_number = n - pn_number - pp_number

    assert len(positive_data) >= pn_number + 2 * pp_number, f"positive data is not enough, expected at least {pn_number + 2 * pp_number}, got {len(positive_data)}"
    assert len(negative_data) >= pn_number + 2 * nn_number, f"negative data is not enough, expected at least {pn_number + 2 * nn_number}, got {len(negative_data)}"

    # build positive-negative data
    xpn_p = positive_data[:pn_number].tolist()
    xpn_n = negative_data[:pn_number].tolist()

    # build positive-positive data
    xpp_p1 = positive_data[pn_number:pn_number+pp_number].tolist()
    xpp_p2 = positive_data[pn_number+pp_number:pn_number+2*pp_number].tolist()

    # build negative-negative data
    xnn_n1 = negative_data[pn_number:pn_number+nn_number].tolist()
    xnn_n2 = negative_data[pn_number+nn_number:pn_number+2*nn_number].tolist()

    x1 = xpn_p + xpp_p1 + xnn_n1
    x2 = xpn_n + xpp_p2 + xnn_n2

    assert len(x1) == len(x2), f"length of x1 and x2 should be equal, got {len(x1)} and {len(x2)}"

    given_y1 = np.concatenate([np.ones(pn_number), np.ones(pp_number), np.ones(nn_number)])
    given_y2 = np.concatenate([-np.ones(pn_number), -np.ones(pp_number), -np.ones(nn_number)])

    real_y1 = np.concatenate([np.ones(pn_number), np.ones(pp_number), -np.ones(nn_number)])
    real_y2 = np.concatenate([-np.ones(pn_number), np.ones(pp_number), -np.ones(nn_number)])

    return x1, x2, given_y1, given_y2, real_y1, real_y2