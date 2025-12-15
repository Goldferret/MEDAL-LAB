#!/bin/bash
# Build extended MADSci Docker image with OpenCV and other dependencies

# Build the image
docker build -t madsci-extended:latest -f Dockerfile.madsci-extended .

echo ""
echo "Image built successfully!"
echo ""
echo "To use this image, update your madsci alias to use 'madsci-extended:latest' instead of 'ghcr.io/ad-sdl/madsci'"
echo ""
echo "Example:"
echo "  Change: ghcr.io/ad-sdl/madsci"
echo "  To:     madsci-extended:latest"

