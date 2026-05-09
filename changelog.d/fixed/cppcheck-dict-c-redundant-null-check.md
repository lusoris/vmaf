- Fixed cppcheck `nullPointer` false-positive in `libvmaf/src/dict.c:121`
  by removing a redundant `&& val` guard inside `dict_overwrite_existing`
  — `val` is already checked at the public entry-point
  `vmaf_dictionary_set` (line 137). Unblocks every PR's CI on
  `Cppcheck (Whole Project)`.
