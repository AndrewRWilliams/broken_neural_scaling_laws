import matplotlib.pyplot as plt
import scipy.optimize
import time
import math

plt.style.use("seaborn-whitegrid")
import numpy as np

GREEN = [0.0, 0.925, 0.0]

"""
Code to reproduce Figure 5 Left of arxiv.org/abs/2210.14891
"""


def bnsl_with_1_break(
    scaling_qtity,
    performance_limit,
    log_log_offset,
    first_slope,
    slope_difference,
    scaling_break,
    break_sharpness,
):
    """broken negative slope log with 1 break"""
    predicted_metric = performance_limit + log_log_offset * scaling_qtity ** (
        -first_slope
    ) * (1 + (scaling_qtity / scaling_break) ** (1 / break_sharpness)) ** (
        -slope_difference * break_sharpness
    )
    return predicted_metric


# TODO: recursively calculate
# def bnsl_with_n_breaks(
#     num_breaks,
#     scaling_qtity,
#     performance_limit,
#     log_log_offset,
#     first_slope,
#     slope_differences,
#     scaling_breaks,
#     break_sharpnesses,
# ):
#     """broken negative slope log with n breaks
#     n is the number of breaks
#     performance_limit is the intercept
#     b is the offset in a log space
#     c0 is the scaling exponent
#     c is a list of n scaling exponents
#     d is a list of n scaling break points along the axis of the scaling quantity
#     f is a list of n scaling break point exponents"""
#     predicted_metric = performance_limit + log_log_offset * scaling_qtity ** (
#         -first_slope
#     ) * (1 + (scaling_qtity / d1) ** (1 / f1)) ** (-c1 * f1)
#     return predicted_metric


def bnsl_with_1_break__log(
    _x,
    performance_limit,
    log_log_offset,
    first_slope,
    slope_difference,
    scaling_break,
    break_sharpness,
):
    """log of bnsl_with_1_break"""
    predicted_metric = bnsl_with_1_break(
        _x,
        performance_limit,
        log_log_offset,
        first_slope,
        slope_difference,
        scaling_break,
        break_sharpness,
    )
    return np.log(predicted_metric + 1)


def bnsl_with_1_break__msle_optim(parameters, _x, _y):
    """mean squared log error"""
    (
        performance_limit,
        log_log_offset,
        first_slope,
        slope_difference,
        scaling_break,
        break_sharpness,
    ) = parameters

    log_log_offset = 1.25**log_log_offset - 1 + 1e-8
    scaling_break = 1.25**scaling_break - 1 + 1e-8

    predicted_metric = bnsl_with_1_break(
        _x,
        performance_limit,
        log_log_offset,
        first_slope,
        slope_difference,
        scaling_break,
        break_sharpness,
    )
    return np.mean((np.log(predicted_metric + 1) - np.log(_y + 1)) ** 2)


def bnsl_with_1_break__sle(parameters, scaling_qtity, true_metric):
    """squared log error"""
    (
        performance_limit,
        log_log_offset,
        first_slope,
        slope_difference,
        scaling_break,
        f1,
    ) = parameters
    predicted_metric = bnsl_with_1_break(
        scaling_qtity,
        performance_limit,
        log_log_offset,
        first_slope,
        slope_difference,
        scaling_break,
        f1,
    )
    return (np.log(predicted_metric) - np.log(true_metric)) ** 2


scaling_points = np.array(
    [
        160,
        192,
        256,
        320,
        384,
        448,
        480,
        512,
        544,
        576,
        608,
        640,
        672,
        736,
        800,
        864,
        928,
    ]
)

metric_points = np.array(
    [
        2.13809046,
        2.11813418,
        2.08955508,
        2.06988398,
        2.05404987,
        2.03837089,
        2.02814281,
        2.00496872,
        1.95576149,
        1.86313841,
        1.70891537,
        1.50637664,
        1.29754721,
        0.96559684,
        0.75856477,
        0.64768338,
        0.55695445,
    ]
)

if __name__ == "__main__":
    print("scaling quantity ground_truth: ", scaling_points)
    print("performance metric ground_truth: ", metric_points)

    fit_comparison_split = 14

    scaling_points_train = scaling_points[:fit_comparison_split]
    metric_points_train = metric_points[:fit_comparison_split]

    scaling_points_comparison = scaling_points[fit_comparison_split:]
    metric_points_comparison = metric_points[fit_comparison_split:]

    plt.plot(
        scaling_points_comparison,
        metric_points_comparison,
        "o",
        color=GREEN,
    )
    plt.plot(
        scaling_points_train,
        metric_points_train,
        "o",
        color="black",
    )

    # grid search range and resolution
    parameter_search_grid = (
        slice(0.0, 1.0, 0.1),
        slice(0, 40, 2.5),
        slice(0, 1, 0.25),
        slice(0, 1, 0.25),
        slice(0, 40, 2.5),
        slice(0, 1, 0.25),
    )

    start = time.time()

    initial_parameters = scipy.optimize.brute(
        bnsl_with_1_break__msle_optim,
        parameter_search_grid,
        args=(scaling_points_train, metric_points_train),
        full_output=False,
        finish=None,
        Ns=1,
        workers=-1,
    )

    [
        performance_limit,
        log_log_offset,
        first_slope,
        slope_difference,
        scaling_break,
        break_sharpness,
    ] = initial_parameters

    # What is this?
    log_log_offset = 1.25**log_log_offset - 1 + 1e-8
    scaling_break = 1.25**scaling_break - 1 + 1e-8
    log_metric_points_train = np.log(metric_points_train + 1)

    optimal_parameters, _ = scipy.optimize.curve_fit(
        bnsl_with_1_break__log,
        scaling_points_train,
        log_metric_points_train,
        p0=initial_parameters,
        maxfev=100000000,
    )

    (
        performance_limit,
        log_log_offset,
        first_slope,
        slope_difference,
        scaling_break,
        break_sharpness,
    ) = optimal_parameters

    total_time = time.time() - start
    print("time: ", total_time)

    points = 4096
    scaling_qtity_resolution = np.array(
        [1.01**i * 10**0 for i in range(points)]
    ).astype(float)

    print("a =", performance_limit)
    print("b =", log_log_offset)
    print("c0 =", first_slope)
    print("c1 =", slope_difference)
    print("d1 =", scaling_break)
    print("f1 =", break_sharpness)

    metric_predictions = bnsl_with_1_break(
        scaling_qtity_resolution.astype(float),
        performance_limit,
        log_log_offset,
        first_slope,
        slope_difference,
        scaling_break,
        break_sharpness,
    )

    plt.plot(
        scaling_qtity_resolution,
        metric_predictions,
        color=[1.0, 0.125, 0.125],
        linewidth=2.5,
    )

    square_log_error = bnsl_with_1_break__sle(
        (
            performance_limit,
            log_log_offset,
            first_slope,
            slope_difference,
            scaling_break,
            break_sharpness,
        ),
        scaling_points,
        metric_points,
    )

    print(
        "root mean square log error, training points: ",
        np.sqrt(np.mean(square_log_error[:fit_comparison_split])),
    )
    print(
        "root mean square log error extrapolate: ",
        np.sqrt(np.mean(square_log_error[fit_comparison_split:])),
    )

    plt.title("4 Digit Addition")
    plt.xlabel("Training Dataset Size")
    plt.ylabel("Test Cross-Entropy")

    """
    plt.xscale('log')
    plt.yscale('log')
    """

    plt.xlim(140, 983)
    plt.ylim(0, 2.5)
    plt.savefig(
        "plot__bnsl__fit_and_extrapolate__4_digit_addition__dataset_size_x-axis_test.png",
        bbox_inches="tight",
    )

    plt.show()

    plt.close()
    plt.cla()
    plt.clf()
