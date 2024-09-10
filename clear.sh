#!/bin/bash

# Usage: ./clear.sh [option]

# Variables
CACHE_CLEAR_COMMAND="rm -rf ~/.cache/ragas"

# Function to show usage
usage() {
    echo "Usage: $0 {cache}"
    exit 1
}

# Function to clear cache
clear_cache() {
    echo "Clearing Ragas cache..."
    $CACHE_CLEAR_COMMAND
    if [ $? -eq 0 ]; then
        echo "Cache cleared successfully!"
    else
        echo "Failed to clear cache."
    fi
}

# Main
if [ "$#" -ne 1 ]; then
    usage
fi

case "$1" in
    cache)
        clear_cache
        ;;
    *)
        usage
        ;;
esac
