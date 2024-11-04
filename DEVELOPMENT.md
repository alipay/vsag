# VSAG developer guide

## Development Environment
There are two ways to build and develop the VSAG project now.

### Docker(recommended)
```bash
docker pull vsaglib/vsag:ubuntu
```

### Dependencies
```bash
# for Debian/Ubuntu
$ ./scripts/deps/install_deps_ubuntu.sh

# for CentOS/AliOS
$ ./scripts/deps/install_deps_centos.sh
```

## VSAG Build Tool
VSAG project use the Unix Makefiles to compile, package and install the library. Here is the commands below:
```bash
Usage: make <target>

Targets:
help:                   ## Show the help.
debug:                  ## Build vsag with debug options.
release:                ## Build vsag with release options.
distribution:           ## Build vsag with distribution options.
fmt:                    ## Format codes.
test:                   ## Build and run unit tests.
asan:                   ## Build with AddressSanitizer option.
test_asan: asan         ## Run unit tests with AddressSanitizer option.
tsan:                   ## Build with ThreadSanitizer option.
test_tsan: tsan         ## Run unit tests with ThreadSanitizer option.
test_cov: cov           ## Build and run unit tests with code coverage enabled.
clean:                  ## Clear build/ directory.
install:                ## Build and install the release version of vsag.
```

## Project Structure
- `benchs/`: benchmark script in Python
- `cmake/`: cmake util functions
- `docker/`: the dockerfile to build develop and ci image
- `examples/`: cpp and python example codes
- `externs/`: third-party libraries
- `include/`: export header files
- `mockimpl/`: the mock implementation that can be used in interface test
- `python_bindings/`: the python bindings
- `scripts/`: useful scripts
- `src/`: the source codes and unit tests
- `tests/`: the functional tests
