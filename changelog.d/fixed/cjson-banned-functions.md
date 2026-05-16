- Replace 11 banned-function call sites (`sprintf`, `strcpy`) in vendored
  `cJSON.c` (v1.7.18) with `snprintf`/`memcpy` equivalents. All replacements
  are bounds-checked against statically-known buffer sizes or the printer's
  `ensure()` contract. No behaviour change. (ADR-0452)
