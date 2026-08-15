# tv-detect build + cross-compile

BINARIES := tv-detect tv-detect-train-logo tv-detect-nn-smoke
BUILD_DIR := build

LDFLAGS := -s -w  # strip debug + symbol tables — smaller binary
GOFLAGS := -trimpath

.PHONY: all build build-all darwin-arm64 linux-arm64 linux-amd64 test clean install ffmpeg-master rebuild ocr

all: build

# Refresh the Mac's ffmpeg/ffprobe to the current FFmpeg master (Homebrew
# --HEAD). Rebuilds only when master actually moved. Kept OUT of `build` so a
# routine `make build` stays fast — run `make rebuild` (or this target) to also
# bump ffmpeg.
ffmpeg-master:
	@./scripts/refresh-ffmpeg-master.sh

# Full detect rebuild: bump ffmpeg to master, then rebuild the binaries.
rebuild: ffmpeg-master build

build: ocr
	@for b in $(BINARIES); do \
		echo "go build $$b (native)"; \
		go build $(GOFLAGS) -ldflags '$(LDFLAGS)' -o $(BUILD_DIR)/$$b ./cmd/$$b || exit 1; \
	done

# Bildschirm-Text-Helfer. Nur macOS: das Vision-Framework ist der Grund,
# warum das ueberhaupt bezahlbar ist (32 ms/Frame, kein Modell-Download,
# kein Netz). Auf Linux entfaellt der Helfer stillschweigend — tv-detect
# laeuft dann ohne das OCR-Signal weiter, siehe internal/signals/ocr.go.
ocr:
	@if [ "$$(uname)" = "Darwin" ] && command -v swiftc >/dev/null 2>&1; then \
		echo "swiftc tv-ocr"; \
		swiftc -O -o $(BUILD_DIR)/tv-ocr tools/ocr/ocr.swift || exit 1; \
	else \
		echo "tv-ocr uebersprungen (kein macOS/swiftc)"; \
	fi

# Cross-compile every target tv-detect actually deploys to.
build-all: darwin-arm64 linux-arm64 linux-amd64

darwin-arm64:
	@for b in $(BINARIES); do \
		GOOS=darwin GOARCH=arm64 go build $(GOFLAGS) -ldflags '$(LDFLAGS)' \
			-o $(BUILD_DIR)/$$b-darwin-arm64 ./cmd/$$b || exit 1; \
	done

linux-arm64:
	@for b in $(BINARIES); do \
		GOOS=linux GOARCH=arm64 go build $(GOFLAGS) -ldflags '$(LDFLAGS)' \
			-o $(BUILD_DIR)/$$b-linux-arm64 ./cmd/$$b || exit 1; \
	done

linux-amd64:
	@for b in $(BINARIES); do \
		GOOS=linux GOARCH=amd64 go build $(GOFLAGS) -ldflags '$(LDFLAGS)' \
			-o $(BUILD_DIR)/$$b-linux-amd64 ./cmd/$$b || exit 1; \
	done

test:
	go test ./...

clean:
	rm -rf $(BUILD_DIR)

# Symlink the dev binaries into PATH (assumes /usr/local/bin is writable).
install: build
	@for b in $(BINARIES); do \
		ln -sf "$(PWD)/$(BUILD_DIR)/$$b" /usr/local/bin/$$b; \
	done
