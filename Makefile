# Makefile

EXE=d2q9-bgk

CC=gcc
# CFLAGS= -std=c99 -Wall -Ofast -mtune=native -fno-tree-vectorize
# CFLAGS= -std=c99 -Wall -Ofast -mtune=native -ftree-vectorize -fopt-info-vec-missed
CFLAGS= -std=c99 -Wall -Ofast -ftree-vectorize -fopt-info-vec-optimized -fopt-info-vec-missed
# CFLAGS= -std=c99 -Wall -Ofast -march=haswell -fopt-info-vec-missed -funsafe-math-optimizations
LIBS = -lm

FINAL_STATE_FILE=./final_state.dat
AV_VELS_FILE=./av_vels.dat
REF_FINAL_STATE_FILE=check/128x128.final_state.dat
REF_AV_VELS_FILE=check/128x128.av_vels.dat

all: $(EXE)

$(EXE): $(EXE).c
	$(CC) $(CFLAGS) $^ $(LIBS) -o $@

check:
	python3 check/check.py --ref-av-vels-file=$(REF_AV_VELS_FILE) --ref-final-state-file=$(REF_FINAL_STATE_FILE) --av-vels-file=$(AV_VELS_FILE) --final-state-file=$(FINAL_STATE_FILE)

.PHONY: all check clean

clean:
	rm -f $(EXE)
