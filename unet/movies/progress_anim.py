import os
"""
    Script to produce a movie out of images in a timeseries.
    Can be used to visualize training progress.
"""
os.system("ffmpeg -f image2 -r 50 -i ../run_1/live_output/p_0_%d.png -vcodec mpeg4 -y ./progress_0.mp4")
os.system("ffmpeg -f image2 -r 50 -i ../run_1/live_output/p_13_%d.png -vcodec mpeg4 -y ./progress_13.mp4")
os.system("ffmpeg -f image2 -r 50 -i ../run_1/live_output/p_26_%d.png -vcodec mpeg4 -y ./progress_26.mp4")
os.system("ffmpeg -f image2 -r 50 -i ../run_1/live_output/e_%d.png -vcodec mpeg4 -y ./descent.mp4")