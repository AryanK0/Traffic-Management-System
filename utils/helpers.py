"""
Small shared helpers.
"""


def clamp(value, lo, hi):

    return max(lo, min(hi, value))


def scale_region(region, from_wh, to_wh):
    """Scale an (x1, y1, x2, y2) region between resolutions."""

    fx = to_wh[0] / float(from_wh[0])
    fy = to_wh[1] / float(from_wh[1])
    x1, y1, x2, y2 = region

    return (int(x1 * fx), int(y1 * fy), int(x2 * fx), int(y2 * fy))


def point_in_region(point, region):

    x, y = point
    x1, y1, x2, y2 = region

    return x1 <= x <= x2 and y1 <= y <= y2


def format_seconds(seconds):

    m, s = divmod(int(seconds), 60)
    return "%d:%02d" % (m, s)
