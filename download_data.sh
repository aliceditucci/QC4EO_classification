#!/bin/bash/

rm -r EuroSAT
rm -r EuroSAT-split

echo "Downloading EuroSAT"
wget --no-check-certificate https://madm.dfki.de/files/sentinel/EuroSAT.zip

echo "Unzip folder and prepare"
unzip -qq EuroSAT.zip
mv 2750 EuroSAT/

# Default split ratio (80% train, 20% validation)
SPLIT_RATIO=80
INPUT_DIR="EuroSAT"
OUTPUT_DIR="EuroSAT-split"
TRAIN_DIR="$OUTPUT_DIR/train"
VAL_DIR="$OUTPUT_DIR/val"

# Create output directories
mkdir -p "$TRAIN_DIR" "$VAL_DIR"

# Loop through each class folder
for class in "$INPUT_DIR"/*; do
    if [ -d "$class" ]; then
        class_name=$(basename "$class")
        mkdir -p "$TRAIN_DIR/$class_name" "$VAL_DIR/$class_name"
        
        # Get all image files and shuffle them
        files=("$class"/*)
        total_files=${#files[@]}
        #train_count=$(echo "$total_files * $SPLIT_RATIO" | bc | awk '{print int($1+0.5)}')
        #train_count=$(echo "$total_files * $SPLIT_RATIO" | bc | awk '{print int($1)}')
        train_count=$(( (total_files * $SPLIT_RATIO) / 100 ))  # For 80% split

        # Shuffle and split files
        shuffled_files=($(printf "%s\n" "${files[@]}" | shuf))
        
        # Move files to respective directories
        for i in "${!shuffled_files[@]}"; do
            if [ "$i" -lt "$train_count" ]; then
                mv "${shuffled_files[$i]}" "$TRAIN_DIR/$class_name/"
            else
                mv "${shuffled_files[$i]}" "$VAL_DIR/$class_name/"
            fi
        done
    fi
done


echo "Dataset split complete:"
echo "Training set: $TRAIN_DIR"
echo "Validation set: $VAL_DIR"