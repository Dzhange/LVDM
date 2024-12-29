#!/bin/bash

FILE_ID="1xWLiU-MBGN7MrsFHQm4_yXmfHBsMbJQo"
FILE_NAME="sky_timelapse.zip"

# Step 1: Get the confirmation token
CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies \
"https://drive.usercontent.google.com/download?export=download&id=${FILE_ID}" -O- \
| grep -oP 'confirm=\K[A-Za-z0-9_]+')

# Step 2: Download the file using the confirmation token
wget --load-cookies /tmp/cookies.txt "https://drive.usercontent.google.com/download?export=download&confirm=${CONFIRM}&id=${FILE_ID}" -O ${FILE_NAME}

# Cleanup
rm /tmp/cookies.txt confirm.txt
