# Included from libvmaf/src/meson.build via `subdir('mcp')`. Paths
# below are relative to libvmaf/src/mcp/.
#
# T5-2 scaffold (ADR-0209) shipped mcp.c only with `-ENOSYS`.
# T5-2b runtime (this PR — see ADR-0209 § "Status update 2026-05-08")
# adds dispatcher.c + transport_stdio.c + a vendored cJSON
# (3rdparty/cJSON/, MIT). mongoose vendoring stays deferred to v2.

mcp_sources = files(
    'mcp.c',
    'dispatcher.c',
    'transport_stdio.c',
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
