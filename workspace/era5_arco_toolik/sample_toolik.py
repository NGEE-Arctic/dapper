"""Run the Toolik watershed through the ERA5-Land ARCO/ELM workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd

from dapper import Domain, ERA5Adapter, sample_era5_land


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("watershed", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--backend", choices=("auto", "arco", "gee"), default="auto")
    parser.add_argument("--start-date", default="1950-01-02")
    parser.add_argument("--end-date", default="latest")
    args = parser.parse_args()

    source = gpd.read_file(args.watershed).to_crs("EPSG:4326")
    support = gpd.GeoDataFrame(
        {
            "gid": ["tfs"],
            "source_file": [args.watershed.name],
            "feature_count": [len(source)],
            "geometry": [source.geometry.union_all()],
        },
        crs="EPSG:4326",
    )
    domain = Domain.from_provided(support, name="tfs", mode="sites")

    csv_dir = args.output / "csv"
    sampled = sample_era5_land(
        domain,
        args.start_date,
        args.end_date,
        backend=args.backend,
        output_dir=csv_dir,
        overwrite=True,
    )
    sampled.export_met(
        src_path=csv_dir,
        adapter=ERA5Adapter(),
        out_dir=args.output / "elm",
        filename="ERA5",
        overwrite=True,
        pack_scope="per-site",
        calendar="noleap",
        dtime_resolution_hrs=1,
        dtime_units="days",
        dformat="BYPASS",
        clip_to_full_years=False,
    )


if __name__ == "__main__":
    main()
