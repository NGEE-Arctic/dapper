# Declarative knobs per source. Start with ERA5; others can copy the shape.

ERA5 = dict(
    id_fallbacks=["gid","pids","pid","id"],
    note="ERA5-Land hourly via GEE",
)
