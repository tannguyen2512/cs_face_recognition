# FACE RECOGNITION PROJECT

## Goals

## Tasks

## Problems - solutions
- Face on edge of webcam: YOLO return the detection coordinate of the result in range `[0,1]`.
Thus, the actual coordinate must be calculated (e.g. `topleft_x = center_x - 0.5*box_width`). When the face is on the edge, such calculation can lead to invalide coordinate (e.g. `topleft_x < 0`). 

    - This can be solved by taking the max, for example, `max(topleft_x, 0)`
