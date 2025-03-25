
### conda activate malstroem

### malstroem pourpts -bluespots D:\Output\bluespots.tif -depths D:\Output\bs_depths.tif -watersheds D:\Output\watersheds.tif -dem D:\Output\Hanwen.tif -out D:\Output\results.gpkg -layername pourpoints -format gpkg

### malstroem pourpts -bluespots D:\Output\bluespots.tif -depths D:\Output\bs_depths.tif -watersheds D:\Output\watersheds.tif -dem D:\Output\Hanwen.tif -out D:\Output\results.gpkg -layername pourpoints -format gpkg

Calculate all derived data for 20mm rain incident ignoring bluespots where the maximum water depth is less than 5cm
and using 20cm statistics resolution when approximating water level of partially ﬁlled bluespots:
### malstroem complete -mm 36.19 -filter "maxdepth > 0.05" -dem D:\Output\Changsha_float32.tif -outdir D:\Output\Changsha -zresolution 0.2

### malstroem complete -mm 20 -filter "volume > 0.5" -dem D:\Output\Hanwen1.tif -outdir D:\Output\222 -zresolution 0.2
