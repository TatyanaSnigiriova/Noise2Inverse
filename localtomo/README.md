https://github.com/pierrepaleo/localtomo/blob/a6aaf373ac9eaa6f7c126fbbca3e178fc44d5052/tomography.py
## Implementation of a new local tomography reconstruction method

This repository contains an implementation of a new method of local tomography reconstruction
based on known sub-region. It is an iterative reconstruction refinement.

The [ASTRA toolbox](https://github.com/astra-toolbox/astra-toolbox/) is required for this implementation to work.

### Usage

The implementation should work out of the box:

```bash
python local1.py
```

This file should bear enough comments to understand the different steps of the method.