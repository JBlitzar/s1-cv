#!/bin/bash


for file in *.mov *.mp4; do
    if [[ -f "$file" ]]; then
        basename="${file%.*}"
        ffmpeg -i "$file" -vf "fps=10,scale=320:-1:flags=lanczos" -vcodec libwebp -lossless 0 -compression_level 6 -q:v 70 -loop 0 -preset picture -an -vsync 0 "${basename}.webp"
        
    fi
done

