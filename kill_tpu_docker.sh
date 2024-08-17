#!/bin/bash

# Find the container ID(s) for the image that starts with "tpuvm"
container_ids=$(sudo docker ps --filter "ancestor=tpuvm" --format "{{.ID}}")

# Check if any container IDs were found
if [ -z "$container_ids" ]; then
    echo "No running containers found with an image that starts with 'tpuvm'."
else
    # Kill each container found
    for id in $container_ids; do
        echo "Killing container with ID: $id"
        sudo docker stop $id
        sudo docker rm $id
    done
fi
