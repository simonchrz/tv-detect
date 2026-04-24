# tv-detect build + cross-compile

BINARIES := tv-detect tv-detect-train-logo tv-detect-nn-smoke
BUILD_DIR := build

LDFLAGS := -s -w  # strip debug + symbol tables — smaller binary
GOFLAGS := -trimpath

.PHONY: all build build-all darwin-arm64 linux-arm64 linux-amd64 test clean install

all: build

build:
	@for b in $(BINARIES); do \
		echo "go build $$b (native)"; \
		go build $(GOFLAGS) -ldflags '$(LDFLAGS)' -o $(BUILD_DIR)/$$b ./cmd/$$b || exit 1; \
	done

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
