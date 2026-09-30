# The frozen extraction code, verbatim

These two files are the exact code that extracted every feature in
`data/upstream_features.csv`. They were frozen before any evaluation ran, and
their SHA-256 fingerprints were recorded at freeze time:

```
18c2db6051ca971cd2d5eac4fd1480ba332038bbc0746c8ef7700a8308b10de7  phi.py
773fe3c3841357c6918c25a451006bf684bd4b55f05df16fad9fec925998b544  pipeline.py
```

Same list in `SHA256SUMS` next to them — `sha256sum -c SHA256SUMS` should say OK
twice. If it doesn't, the files you have are not the files the paper used.

Fair warning: this is working research code, kept exactly as it ran (terse names,
comments in Chinese, hash self-asserts). It is here as evidence, not as the thing
to read. The readable version is `code/01_collect_and_build_features/`, which
reimplements this logic function by function and was checked against it on
synthetic cones — identical outputs, byte for byte. If the two ever disagree,
these frozen files win.
