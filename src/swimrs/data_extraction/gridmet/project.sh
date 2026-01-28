#!/bin/bash
src_epsg="EPSG:4326"
dst_epsg="EPSG:5071"
input_directory="/media/research/IrrigationGIS/et-demands/gridmet/gridmet_corrected/correction_surfaces_wgs"
output_directory="/media/research/IrrigationGIS/et-demands/gridmet/gridmet_corrected/gridmet_surfaces_aea"

# Create the output directory if it doesn't exist
mkdir -p "$output_directory"

# Loop through each raster file in the input directory and transform it
for input_file in "$input_directory"/*.tif; do
  output_file="$output_directory/$(basename "$input_file")"
  gdalwarp -s_srs "$src_epsg" -t_srs "$dst_epsg" "$input_file" "$output_file"
done
