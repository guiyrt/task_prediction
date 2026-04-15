#!/bin/bash

# --- Configuration ---
REMOTE_HOST="tern@192.168.68.64"
REMOTE_SOURCE_DIR="/home/tern/zhaw/data"
LOCAL_DEST_DIR="/home/regd/AWARE/april26"
ABORTED_SUBDIR="aborted"

# Matches YYYYMMDD__HHMMSS (8 digits, 2 underscores, 6 digits)
TIMESTAMP_REGEX="__[0-9]{8}_[0-9]{6}"
IGNORE_DIR=".calibrations"
DRY_RUN=${DRY_RUN:-false}

# Rsync flags
# -a: archive, -v: verbose, -z: compress, -P: progress, r: recursive
RSYNC_OPTS="-avzPr"
[ "$DRY_RUN" = "true" ] && RSYNC_OPTS="$RSYNC_OPTS --dry-run"

# --- Execution ---
echo "Connecting to $REMOTE_HOST to list folders..."

# Get all available directories
REMOTE_DIRS=$(ssh "$REMOTE_HOST" "find '$REMOTE_SOURCE_DIR' -maxdepth 1 -mindepth 1 -type d -not -name '$IGNORE_DIR' -printf '%f\n'")

if [ -z "$REMOTE_DIRS" ]; then
    echo "No directories found on remote host at $REMOTE_SOURCE_DIR"
    exit 0
fi

# Create local temp files for the rsync lists
CLEAN_LIST=$(mktemp)
ABORTED_LIST=$(mktemp)

# Categorize the names locally (successful vs aborted runs)
while read -r dir_name; do
    if echo "$dir_name" | grep -Eq "$TIMESTAMP_REGEX"; then
        echo "$dir_name" >> "$ABORTED_LIST"
    else
        echo "$dir_name" >> "$CLEAN_LIST"
    fi
done <<< "$REMOTE_DIRS"

# Perform the Pulling
if [ "$DRY_RUN" = "true" ]; then
    echo "--- DRY RUN ENABLED ---"
fi

# Pull Clean Folders (Successes)
if [ -s "$CLEAN_LIST" ]; then
    # Create local folder if doesn't exist
    mkdir -p "${LOCAL_DEST_DIR}"

    echo "--- Pulling SUCCESS scenarios to $LOCAL_DEST_DIR ---"
    rsync $RSYNC_OPTS --files-from="$CLEAN_LIST" "$REMOTE_HOST:$REMOTE_SOURCE_DIR" "$LOCAL_DEST_DIR"
fi

# Pull Aborted Folders (Failures)
if [ -s "$ABORTED_LIST" ]; then
    # Create the local aborted directory if it doesn't exist
    mkdir -p "${LOCAL_DEST_DIR}/${ABORTED_SUBDIR}"
    
    echo "--- Pulling ABORTED scenarios to ${LOCAL_DEST_DIR}/${ABORTED_SUBDIR} ---"
    rsync $RSYNC_OPTS --files-from="$ABORTED_LIST" "$REMOTE_HOST:$REMOTE_SOURCE_DIR" "${LOCAL_DEST_DIR}/${ABORTED_SUBDIR}/"
fi

# Cleanup
rm "$CLEAN_LIST" "$ABORTED_LIST"

if [ "$DRY_RUN" = "true" ]; then
    echo "--- Dry Run Complete. Run with 'DRY_RUN=false ./copy_from_polaris.sh' to execute. ---"
fi