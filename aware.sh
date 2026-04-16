#!/bin/bash

# ==============================================================================
# Script: manage_aware.sh
# Purpose: Unified tool for Syncing, Repairing MKVs, and Docker Deployment.
# ==============================================================================

set -euo pipefail

# --- Configuration: Hosts ---
readonly POLARIS_HOST="tern@192.168.68.64"
readonly ORION_HOST="tern@192.168.68.69"
readonly DEPLOY_HOST="$POLARIS_HOST"

# --- Configuration: Paths ---
readonly POLARIS_SRC="/home/tern/zhaw/data"
readonly POLARIS_DEST="/home/regd/AWARE/april26"
readonly ORION_SRC="/home/tern/git/polaris-aware/event-recordings"
readonly ORION_DEST="/home/regd/AWARE/april26/orion"

# --- Configuration: Docker Services ---
readonly SERVICES=(
    "$HOME/task_prediction:task-pred"
    "$HOME/instance_pred:instance-pred"
    "$HOME/screen_recording:screen-recorder"
    "$HOME/gaze_capture:gaze-capture"
)

# --- Configuration: Logic ---
readonly TIMESTAMP_REGEX="__[0-9]{8}_[0-9]{6}"
readonly DRY_RUN=${DRY_RUN:-false}

# --- Formatting ---
log_info()    { echo -e "\033[1;34m[INFO]\033[0m  $*"; }
log_success() { echo -e "\033[1;32m[OK]\033[0m    $*"; }
log_fix()     { echo -e "\033[1;35m[REPAIR]\033[0m $*"; }
log_error()   { echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }

# ==============================================================================
# 1. DATA REPAIR ENGINE
# ==============================================================================

repair_mkv_files() {
    local search_path="$1"
    [[ "$DRY_RUN" == "true" ]] && return
    
    log_info "Scanning for broken MKVs in $search_path..."
    
    # Use FD 3 to avoid stdin conflicts with ffmpeg
    while IFS= read -r -d '' file <&3; do
        
        # Check if file is broken. Redirect stderr to /dev/null to hide the 
        # "File ended prematurely" warning during the check phase.
        if ! ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$file" 2>/dev/null | grep -qE "^[0-9]"; then
            
            local temp_file="${file}.tmp.mkv"
            log_fix "Standardizing headers: $(basename "$file")"
            
            # -nostdin is vital to prevent the loop from breaking
            if ffmpeg -nostdin -loglevel error -y -i "$file" -c copy -f matroska "$temp_file"; then
                if [[ -s "$temp_file" ]]; then
                    mv "$temp_file" "$file"
                else
                    rm -f "$temp_file"
                fi
            else
                log_error "Failed to repair $file"
                rm -f "$temp_file"
            fi
        fi
    done 3< <(find "$search_path" -type f -name "*.mkv" -print0)
}

# ==============================================================================
# 2. SYNC ENGINE
# ==============================================================================

run_sync() {
    local label="$1" host="$2" src="$3" dest="$4"; shift 4
    log_info "Syncing $label..."
    mkdir -p "$dest"
    
    local opts=("-avzPr" "--ignore-existing")
    [[ "$DRY_RUN" == "true" ]] && opts+=("--dry-run")
    
    rsync "${opts[@]}" "$@" "$host:$src/" "$dest/"
}

# ==============================================================================
# 3. DEPLOY ENGINE
# ==============================================================================

deploy_containers() {
    log_info "Starting Docker Deployment to $DEPLOY_HOST..."
    for service in "${SERVICES[@]}"; do
        IFS=":" read -r s_dir s_tag <<< "$service"
        [[ ! -d "$s_dir" ]] && continue

        log_info "Building & Streaming: $s_tag"
        (
            cd "$s_dir"
            docker build --platform linux/amd64 -t "$s_tag:latest" .
            docker save "$s_tag:latest" | ssh "$DEPLOY_HOST" "docker load"
        )
    done
}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

main() {
    local action=${1:-"all"}
    local start_time=$SECONDS

    CLEAN_LIST=$(mktemp)
    ABORTED_LIST=$(mktemp)
    trap 'rm -f "$CLEAN_LIST" "$ABORTED_LIST"' EXIT

    case "$action" in
        "sync")
            # --- Orion ---
            run_sync "Orion DBs" "$ORION_HOST" "$ORION_SRC" "$ORION_DEST" --include="*/" --include="*.db" --exclude="*"

            # --- Polaris ---
            log_info "Querying Polaris ($POLARIS_HOST)..."
            local remote_dirs
            # Filter to ensure we only get the directory names, ignoring terminal noise
            remote_dirs=$(ssh -q "$POLARIS_HOST" "find '$POLARIS_SRC' -maxdepth 1 -mindepth 1 -type d -not -name '.calibrations' -printf '%f\n'" | grep -E '^[0-9]' || true)
            
            if [[ -n "$remote_dirs" ]]; then
                while read -r dir; do
                    [[ -z "$dir" ]] && continue
                    [[ "$dir" =~ $TIMESTAMP_REGEX ]] && echo "$dir" >> "$ABORTED_LIST" || echo "$dir" >> "$CLEAN_LIST"
                done <<< "$remote_dirs"

                [[ -s "$CLEAN_LIST" ]] && run_sync "Polaris Successes" "$POLARIS_HOST" "$POLARIS_SRC" "$POLARIS_DEST" --files-from="$CLEAN_LIST"
                [[ -s "$ABORTED_LIST" ]] && run_sync "Polaris Aborted" "$POLARIS_HOST" "$POLARIS_SRC" "$POLARIS_DEST/aborted" --files-from="$ABORTED_LIST"
                
                # Repair MKVs locally after sync
                repair_mkv_files "$POLARIS_DEST"

                # Generated asd_events from db
                source .venv/bin/activate
                python scripts/process_polaris_db.py "$POLARIS_DEST"
            fi
            ;;

        "deploy")
            deploy_containers
            ;;

        *)
            echo "Usage: $0 {sync|deploy}"
            exit 1
            ;;
    esac

    log_success "Completed in $(( SECONDS - start_time ))s."
}

main "$@"