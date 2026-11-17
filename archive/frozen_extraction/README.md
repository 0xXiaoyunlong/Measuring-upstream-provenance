# The frozen extraction code

These two files are the code that extracted every feature in
`data/upstream_features.csv`. They were frozen before any evaluation ran. Their
comments were written in Chinese and are translated here; the code itself is
untouched.

Fair warning: this is working research code, kept as it ran (terse names, dense
one-liners). It is here as evidence, not as the thing to read. The readable
version is `code/01_collect_and_build_features/`, which reimplements this logic
function by function and was checked against it on synthetic cones — identical
outputs, byte for byte. If the two ever disagree, these frozen files win.
