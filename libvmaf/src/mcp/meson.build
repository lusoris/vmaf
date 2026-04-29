# Included from libvmaf/src/meson.build via `subdir('mcp')`. Paths
# below are relative to libvmaf/src/mcp/.
#
# T5-2 scaffold (ADR-0209): exposes `mcp_sources` + `mcp_defines` for
# the parent meson.build to thread into the libvmaf static-lib. No
# runtime dependencies — cJSON + mongoose vendoring lands with the
# T5-2b runtime PR.

mcp_sources = files(
    'mcp.c',
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
