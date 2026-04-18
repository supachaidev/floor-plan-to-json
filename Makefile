# Floor Plan to JSON — task runner
#
# Usage:
#   make help        list the available targets
#   make setup       install python dependencies
#   make generate    regenerate the reference dataset
#   make eval        run vectorizer accuracy evaluation
#   make convert     lift sample 000 into a RoomPlan CapturedRoom JSON
#   make view        render a 3D preview of the converted sample
#   make demo        generate -> convert -> view in one go
#   make test        prove the convert pipeline is raster-only
#   make clean       remove generated artefacts under .tmp/
#
# Variables (override on the command line):
#   SAMPLE=003       use dataset/floorplan_003.png for convert / view (default: 000)
#   N=20             number of samples to generate (default: 10)
#   SEED=1000        base seed for generator (default: 1000)

PYTHON ?= python3
DATASET ?= dataset
OUT_DIR ?= .tmp
SAMPLE ?= 000
N ?= 10
SEED ?= 1000

SAMPLE_PNG  := $(DATASET)/floorplan_$(SAMPLE).png
ROOM_JSON   := $(OUT_DIR)/room_$(SAMPLE).json
ROOM_PNG    := $(OUT_DIR)/room_$(SAMPLE).png

.DEFAULT_GOAL := help
.PHONY: help setup generate eval evaluate convert view demo test clean

help:
	@echo "Floor Plan to JSON — task runner"
	@echo ""
	@echo "Targets:"
	@echo "  setup      install python dependencies"
	@echo "  generate   regenerate the reference dataset (N=$(N), SEED=$(SEED))"
	@echo "  eval       run vectorizer accuracy evaluation on $(DATASET)/"
	@echo "  convert    lift $(SAMPLE_PNG) into $(ROOM_JSON)"
	@echo "  view       render $(ROOM_JSON) to $(ROOM_PNG)"
	@echo "  demo       convert + view in one go"
	@echo "  test       prove convert works with no paired GT JSON"
	@echo "  clean      remove $(OUT_DIR)/"
	@echo ""
	@echo "Override SAMPLE=NNN to point convert/view at a different sample."

setup:
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install matplotlib

generate:
	$(PYTHON) generator.py -n $(N) -o $(DATASET) --seed $(SEED)

eval evaluate:
	$(PYTHON) main.py evaluate $(DATASET)

$(OUT_DIR):
	mkdir -p $(OUT_DIR)

convert: | $(OUT_DIR)
	$(PYTHON) main.py convert $(SAMPLE_PNG) -o $(ROOM_JSON)

view: | $(OUT_DIR)
	$(PYTHON) main.py view $(ROOM_JSON) -s $(ROOM_PNG)
	@echo "Saved preview to $(ROOM_PNG)"

demo: convert view

test: | $(OUT_DIR)
	@echo ">> Proving convert is raster-only (no paired JSON)"
	@cp $(SAMPLE_PNG) $(OUT_DIR)/blind.png
	@$(PYTHON) main.py convert $(OUT_DIR)/blind.png -o $(OUT_DIR)/blind_room.json
	@$(PYTHON) -c "import json; r=json.load(open('$(OUT_DIR)/blind_room.json')); print(f'  walls={len(r[\"walls\"])}  doors={len(r[\"doors\"])}')"

clean:
	rm -rf $(OUT_DIR)
