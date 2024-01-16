# pygraphblas/Makefile

# DEVELOP with `make debug`
debug: 
	nodemon -L -V

# use "time -p" for a benchmark (pytest has timings, but this time combines multiple sets of tests together)
.PHONY: debug_command
debug-command:
	time -p $(MAKE) test

# TESTS
test: test-parallel
	

# number of worker threads (2 works in github actions, more is good for lots of tests (we don't have a lot))
N_WORKERS=2

# parallel unit tests (for dev rig)
test-parallel:
	py.test -n $(N_WORKERS) --cov=pygraphblas --cov-config=.coveragerc --cov-report=term-missing --cov-report=lcov:coverage/lcov.info -vv tests

# sequential unit tests (for CI)
test-sequential:
	pytest --cov=pygraphblas --cov-config=.coveragerc --cov-report=term-missing --cov-report=lcov:coverage/lcov.info -vv tests