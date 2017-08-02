START_NUM=$1
NUM=$START_NUM
time python labeler.py --video_in /lfs/1/ddkang/noscope/data/videos/taipei-long.mp4 --csv_out /lfs/1/ddkang/noscope/csvs-inception-ssd/taipei-long/taipei-long.$NUM.csv --start_frame $((NUM * 300000))
NUM=$((START_NUM + 12))
time python labeler.py --video_in /lfs/1/ddkang/noscope/data/videos/taipei-long.mp4 --csv_out /lfs/1/ddkang/noscope/csvs-inception-ssd/taipei-long/taipei-long.$NUM.csv --start_frame $((NUM * 300000))
