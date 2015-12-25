#!/bin/bash

./picogen rnt/cascades/facefinder -r 0.0 facedet > rnt/cascades/face-cpu.h
./picogen rnt/cascades/facefinder -r 0.0 facedet --cuda > rnt/cascades/face-cuda.h
