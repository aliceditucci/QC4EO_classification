#!/bin/bash/

rm -r EuroSAT
rm -r EuroSAT-split
rm -f EuroSAT.zip

echo "Downloading EuroSAT"
# RGB
# wget --no-check-certificate https://madm.dfki.de/files/sentinel/EuroSAT.zip 
# MultiSpectral
wget --no-check-certificate https://madm.dfki.de/files/sentinel/EuroSATallBands.zip 

echo "Unzip folder and prepare"

# RGB
#unzip -qq EuroSAT.zip
#mv 2750 EuroSAT/

# Multispectral
unzip -qq EuroSATallBands.zip
mv ds/images/remote_sensing/otherDatasets/sentinel_2/tif EuroSAT/

# Split ratios
TRAIN_RATIO=80
VAL_RATIO=15
TEST_RATIO=5

INPUT_DIR="EuroSAT"
OUTPUT_DIR="EuroSAT-split"
TRAIN_DIR="$OUTPUT_DIR/train"
VAL_DIR="$OUTPUT_DIR/val"
TEST_DIR="$OUTPUT_DIR/test"

# Create output directories
mkdir -p "$TRAIN_DIR" "$VAL_DIR" "$TEST_DIR"

# Loop through each class folder
for class in "$INPUT_DIR"/*; do
    if [ -d "$class" ]; then
        class_name=$(basename "$class")
        mkdir -p "$TRAIN_DIR/$class_name" "$VAL_DIR/$class_name" "$TEST_DIR/$class_name"

        # Get all image files and shuffle them
        files=("$class"/*)
        total_files=${#files[@]}

        train_count=$(( total_files * TRAIN_RATIO / 100 ))
        val_count=$(( total_files * VAL_RATIO / 100 ))
        test_count=$(( total_files - train_count - val_count ))  # Ensure total matches

        shuffled_files=($(printf "%s\n" "${files[@]}" | shuf))

        for i in "${!shuffled_files[@]}"; do
            if [ "$i" -lt "$train_count" ]; then
                mv "${shuffled_files[$i]}" "$TRAIN_DIR/$class_name/"
            elif [ "$i" -lt $((train_count + val_count)) ]; then
                mv "${shuffled_files[$i]}" "$VAL_DIR/$class_name/"
            else
                mv "${shuffled_files[$i]}" "$TEST_DIR/$class_name/"
            fi
        done
    fi
done

echo "Dataset split complete:"
echo "Training set: $TRAIN_DIR"
echo "Validation set: $VAL_DIR"
echo "Test set: $TEST_DIR"