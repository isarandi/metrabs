import numpy as np
import sklearn.dummy
import sklearn.linear_model


def eta_string(time_points, remaining_works):
    return format_timedelta(eta(time_points, remaining_works))


def eta(time_points, remaining_works, regression_points_used=200):
    """Estimate the time remaining until completion of a task based on step history.

    Args:
        time_points: a sequence of points in time, represented as seconds elapsed since a common
            reference, such as the Unix epoch or anything else.
        remaining_works: a sequence of amounts of work that were remaining to be done at each
            time in `time_points`. The values must be non-negative and 0 represents the
            full completion of the task.
        regression_points_used: This many of the last measurments are used in the linear regression.

    Returns:
        The estimated time remaining until completion of the task, relative to the last element
        of `time_points`.
    """
    time_points = np.asarray(time_points)
    remaining_works = np.asarray(remaining_works)
    return np.mean([
        eta_linear_regression_shifted(
            time_points[-regression_points_used:],
            remaining_works[-regression_points_used:]),
        eta_lookback(time_points, remaining_works)])


def eta_lookback(t, r):
    """Before half the work is done, make an estimate based on the first and last data points.
    After that: given X amount of work remaining, assume that this X amount of
    work will take the same time that the most recent X amount of work took.
    In other words, the time remaining is equal to the time that elapsed between having 2*X
    remaining work and X remaining work (i.e. now)."""

    # Until half is done, estimate based on first and last:
    if r[-1] * 2 >= r[0]:
        time_per_unit_work = (t[-1] - t[0]) / (r[0] - r[-1])
        return r[-1] * time_per_unit_work

    # Find the time when twice the current remaining work was remaining, by linear interpolation.
    # np.interp requires an increasing sequence, hence the negative signs.
    t_twice_remaining = np.interp(-r[-1] * 2, -r, t)
    return t[-1] - t_twice_remaining


def eta_linear_regression_shifted(t, r):
    """Estimate the time remaining by the following method:
    Calculate the speed of progress by linear regression then make an estimate considering this
    speed and the current work amount remaining (shifting the regression line
    to go through the last point while keeping the slope)."""

    model = sklearn.linear_model.LinearRegression()
    model.fit(np.expand_dims(t, axis=1), r)
    speed = -model.coef_[0]
    return r[-1] / speed


def format_timedelta(seconds):
    if seconds is None or np.isnan(seconds):
        return 'unknown'
    if seconds == np.inf:
        return '∞'
    if int(seconds) <= 0:
        return '<= 0'

    seconds = int(seconds)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    years, days = divmod(days, 365)

    values = [years, days, hours, minutes, seconds]
    unit_names = ['year', 'day', 'hour', 'minute', 'second']

    # Get first index where the value is larger than 0
    i1 = next(i for (i, x) in enumerate(values) if x > 0)
    return ', '.join(f'{value} {unit_name}{"" if value == 1 else "s"}'
                     for value, unit_name in zip(values[i1:i1 + 2], unit_names[i1:i1 + 2]))
