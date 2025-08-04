#!/bin/bash

# Wavetrainer repository update script
# Author: Automated script for wavetrainer development environment
# Usage: ./update_all.sh

set -e  # Exit on any error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🌊 Starting Wavetrainer update process...${NC}"
echo ""

# Activate conda environment
echo -e "${YELLOW}📦 Activating Sportsball environment...${NC}"
source /opt/homebrew/Caskroom/miniconda/base/etc/profile.d/conda.sh
conda activate Sportsball

# Sync with upstream
echo -e "${YELLOW}🔄 Step 1: Syncing with upstream wavetrainer...${NC}"
git fetch upstream
git fetch origin

# Check if we have uncommitted changes
if ! git diff-index --quiet HEAD --; then
    echo -e "${YELLOW}⚠️ Uncommitted changes detected, stashing...${NC}"
    git stash push -m "Auto-stash before update $(date)"
    STASHED=true
else
    STASHED=false
fi

# Merge upstream changes
echo "Merging upstream/main..."
git merge upstream/main --no-edit

# Restore stashed changes if any
if [ "$STASHED" = true ]; then
    echo -e "${YELLOW}📦 Restoring stashed changes...${NC}"
    git stash pop
fi

# Install wavetrainer in development mode
echo -e "${YELLOW}🔧 Step 2: Installing wavetrainer in development mode...${NC}"
pip install -e .

# Push updated main to your fork
echo -e "${YELLOW}📤 Step 3: Pushing to origin...${NC}"
git push origin main

echo ""
echo -e "${GREEN}🎉 Wavetrainer update completed successfully!${NC}"
echo -e "${BLUE}📋 Final status:${NC}"

echo "✅ Wavetrainer: $(pip show wavetrainer | grep Version | cut -d' ' -f2)"
echo "✅ Current branch: $(git branch --show-current)"
echo "✅ Status vs upstream: $(git rev-list --count --left-right HEAD...upstream/main | tr '\t' ' commits ahead, ') commits behind"
echo "✅ Installation: $(pip show wavetrainer | grep Location | cut -d' ' -f2-)"