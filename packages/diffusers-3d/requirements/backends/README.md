# Backend constraints

This directory records audited source revisions and build constraints for dependencies that cannot be represented by
portable PyPI extras.

Constraint files must include:

- the immutable source commit;
- package and import names;
- license classification;
- supported Python, PyTorch, CUDA/HIP, compiler, and operating-system combinations;
- build-isolation requirements;
- a checksum for any externally hosted wheel.

Do not add mutable branches or unverified third-party wheels.
