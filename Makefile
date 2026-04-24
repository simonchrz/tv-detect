# tv-detect build + cross-compile

BINARY := tv-detect
BUILD_DIR := build
PKG := ./cmd/tv-detect

LDFLAGS := -s -w  # strip debug + symbol tables — smaller binary
GOFLAGS := -trimpath

.PHONY: all build build-all darwin-arm64 linux-arm64 linux-amd64 test clean install

all: build

build:
	go build $(GOFLAGS) -ldflags '$(LDFLAGS)' -o $(BUILD_DIR)/$(BINARY) $(PKG)

# Cross-compile every target tv-detect actually deploys to.
build-all: darwin-arm64 linux-arm64 linux-amd64

darwin-arm64:
	GOOS=darwin GOARCH=arm64 go build $(GOFLAGS) -ldflags '$(LDFLAGS)' \
		-o $(BUILD_DIR)/$(BINARY)-darwin-arm64 $(PKG)

linux-arm64:
	GOOS=linux GOARCH=arm64 go build $(GOFLAGS) -ldflags '$(LDFLAGS)' \
		-o $(BUILD_DIR)/$(BINARY)-linux-arm64 $(PKG)

linux-amd64:
	GOOS=linux GOARCH=amd64 go build $(GOFLAGS) -ldflags '$(LDFLAGS)' \
		-o $(BUILD_DIR)/$(BINARY)-linux-amd64 $(PKG)

test:
	go test ./...

clean:
	rm -rf $(BUILD_DIR)

# Symlink the dev binary into PATH (assumes /usr/local/bin is writable).
install: build
	ln -sf "$(PWD)/$(BUILD_DIR)/$(BINARY)" /usr/local/bin/$(BINARY)
