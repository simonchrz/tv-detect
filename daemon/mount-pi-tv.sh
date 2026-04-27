#!/bin/bash
# Mount the Pi's [tv] SMB share at ~/mnt/pi-tv using the password
# stashed in login-keychain (acct=simon, ptcl=smb, srvr=raspberrypi5lan).
#
# mount_smbfs alone doesn't pull from keychain (only Finder's
# CocoaURL path does), so we grab the password explicitly via
# `security` and feed it into the URL. Required after every Pi
# reboot, or whenever the SMB connection has died.
#
# tv-train-head.sh and tv-comskip.sh both check ~/mnt/pi-tv via
# `mount | grep`, so the path matters — using Finder's default
# /Volumes/tv would skip them.

set -euo pipefail
TARGET="$HOME/mnt/pi-tv"
HOST="raspberrypi5lan"
SHARE="tv"
USER="simon"

if mount | grep -q "on $TARGET ("; then
  echo "already mounted: $TARGET"; exit 0
fi
mkdir -p "$TARGET"

# In case macOS auto-mounted to /Volumes/tv first, drop that.
if mount | grep -q "on /Volumes/$SHARE "; then
  umount "/Volumes/$SHARE" 2>/dev/null || true
fi

PW=$(security find-internet-password -s "$HOST" -a "$USER" -r 'smb ' -w 2>/dev/null) || {
  echo "no SMB password for $USER@$HOST in keychain — mount once via Finder first" >&2
  exit 1
}
PW_ENC=$(python3 -c 'import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1], safe=""))' "$PW")

/sbin/mount_smbfs "//$USER:$PW_ENC@$HOST/$SHARE" "$TARGET"
echo "mounted: $TARGET"
