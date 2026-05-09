# Included from libvmaf/src/meson.build via `subdir('mcp')`. Paths
# below are relative to libvmaf/src/mcp/.
#
# T5-2 scaffold (ADR-0209) shipped mcp.c only with `-ENOSYS`.
# T5-2b runtime (PR #490) added dispatcher.c + transport_stdio.c +
# a vendored cJSON (3rdparty/cJSON/, MIT) + a placeholder
# `compute_vmaf` returning {"status":"deferred_to_v2"}.
# T5-2c (this PR — MCP runtime v2) adds transport_uds.c (Unix
# domain socket) and compute_vmaf.c (real libvmaf scoring binding).
# SSE transport (mongoose vendoring) remains deferred to v3.

mcp_sources = files(
    'mcp.c',
    'dispatcher.c',
    'compute_vmaf.c',
    'transport_stdio.c',
    'transport_uds.c',
    '3rdparty/cJSON/cJSON.c',
)

mcp_defines = ['-DHAVE_MCP=1']

if get_option('enable_mcp_sse')
    mcp_defines += '-DHAVE_MCP_SSE=1'
endif
if get_option('enable_mcp_uds')
    mcp_defines += '-DHAVE_MCP_UDS=1'
endif
if get_option('enable_mcp_stdio')
    mcp_defines += '-DHAVE_MCP_STDIO=1'
endif
