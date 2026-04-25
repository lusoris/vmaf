/**
 *  Copyright 2026 Lusoris and Claude (Anthropic)
 *  SPDX-License-Identifier: BSD-3-Clause-Plus-Patent
 *
 *  Tests for the ONNX-protobuf field scanner (src/dnn/onnx_scan.c).
 *  Buffers are hand-crafted: we don't want to depend on a real protobuf
 *  library to generate fixtures — that would reintroduce the dependency
 *  the scanner was written to avoid.
 *
 *  Tag byte = (field_number << 3) | wire_type
 *      ModelProto.graph    field 7, wire 2  →  0x3A
 *      GraphProto.node     field 1, wire 2  →  0x0A
 *      NodeProto.op_type   field 4, wire 2  →  0x22
 */

#include <errno.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "test.h"

#include "dnn/onnx_scan.h"

static char *test_null_buffer(void)
{
    const int err = vmaf_dnn_scan_onnx(NULL, 10, NULL);
    mu_assert("NULL buffer must return -EINVAL", err == -EINVAL);
    return NULL;
}

static char *test_zero_length(void)
{
    const unsigned char buf[1] = {0};
    const int err = vmaf_dnn_scan_onnx(buf, 0, NULL);
    mu_assert("zero-length must return -EINVAL", err == -EINVAL);
    return NULL;
}

static char *test_no_graph_field(void)
{
    /* One varint field (field 1, value 1) — no ModelProto.graph present. */
    const unsigned char buf[] = {0x08, 0x01};
    const int err = vmaf_dnn_scan_onnx(buf, sizeof(buf), NULL);
    mu_assert("missing graph must return -ENOENT", err == -ENOENT);
    return NULL;
}

static char *test_allowed_op_conv(void)
{
    /* ModelProto { graph = { node = { op_type = "Conv" } } } */
    const unsigned char buf[] = {
        0x3A, 0x08,                    /* ModelProto.graph, len=8    */
        0x0A, 0x06,                    /* GraphProto.node, len=6     */
        0x22, 0x04, 'C', 'o', 'n', 'v' /* NodeProto.op_type = "Conv" */
    };
    char *first_bad = NULL;
    const int err = vmaf_dnn_scan_onnx(buf, sizeof(buf), &first_bad);
    mu_assert("Conv must be accepted", err == 0);
    mu_assert("first_bad must remain NULL on success", first_bad == NULL);
    return NULL;
}

static char *test_allowed_multiple_ops(void)
{
    /* Two nodes: Conv then Relu. */
    const unsigned char buf[] = {0x3A, 0x10, /* ModelProto.graph, len=16   */
                                 0x0A, 0x06, /* node, len=6                */
                                 0x22, 0x04, 'C', 'o', 'n', 'v',
                                 0x0A, 0x06, /* node, len=6                */
                                 0x22, 0x04, 'R', 'e', 'l', 'u'};
    const int err = vmaf_dnn_scan_onnx(buf, sizeof(buf), NULL);
    mu_assert("Conv+Relu must be accepted", err == 0);
    return NULL;
}

static char *test_loop_top_level_allowed(void)
{
    /* ADR-0169 / T6-5: bare Loop at the top level (no body subgraph
     * fixture) is now accepted at the op_type layer. A real Loop would
     * embed a body GraphProto in NodeProto.attribute; that recursion
     * is exercised by `test_loop_with_forbidden_subgraph` /
     * `test_loop_with_allowed_subgraph` below. */
    const unsigned char buf[] = {0x3A, 0x08, 0x0A, 0x06, 0x22, 0x04, 'L', 'o', 'o', 'p'};
    char *first_bad = NULL;
    const int err = vmaf_dnn_scan_onnx(buf, sizeof(buf), &first_bad);
    mu_assert("Loop must be accepted", err == 0);
    mu_assert("first_bad must remain NULL on success", first_bad == NULL);
    return NULL;
}

static char *test_if_after_allowed_now_accepted(void)
{
    /* Same fixture as the historical `test_disallowed_op_if_after_allowed`,
     * but post-ADR-0169 If is on the allowlist. Both nodes accepted. */
    const unsigned char buf[] = {0x3A, 0x0E, /* graph, len=14         */
                                 0x0A, 0x06, 0x22, 0x04, 'C',
                                 'o',  'n',  'v',  0x0A, 0x04, /* node, len=4 */
                                 0x22, 0x02, 'I',  'f'};
    char *first_bad = NULL;
    const int err = vmaf_dnn_scan_onnx(buf, sizeof(buf), &first_bad);
    mu_assert("Conv+If must both be accepted", err == 0);
    mu_assert("first_bad must remain NULL on success", first_bad == NULL);
    return NULL;
}

static char *test_scan_still_rejected(void)
{
    /* Scan stays off the allowlist (ADR-0169 § Alternatives considered). */
    const unsigned char buf[] = {0x3A, 0x08, 0x0A, 0x06, 0x22, 0x04, 'S', 'c', 'a', 'n'};
    char *first_bad = NULL;
    const int err = vmaf_dnn_scan_onnx(buf, sizeof(buf), &first_bad);
    mu_assert("Scan must be rejected", err == -EPERM);
    mu_assert("first_bad must equal \"Scan\"", first_bad && strcmp(first_bad, "Scan") == 0);
    free(first_bad);
    return NULL;
}

/* Hand-crafted ModelProto: ModelProto { graph = { node = Loop with
 * NodeProto.attribute { name="body", type=GRAPH, g={ node=Conv } } } }
 * Layout (31 bytes total):
 *   3A 1D                            ModelProto.graph,    len=29
 *   0A 1B                            GraphProto.node,     len=27 (Loop node)
 *   22 04 4C 6F 6F 70                NodeProto.op_type = "Loop"
 *   2A 13                            NodeProto.attribute, len=19
 *   0A 04 62 6F 64 79                AttributeProto.name = "body"
 *   A0 01 05                         AttributeProto.type = GRAPH (5)
 *   32 08                            AttributeProto.g, len=8
 *   0A 06                            GraphProto.node, len=6 (inner)
 *   22 04 43 6F 6E 76                NodeProto.op_type = "Conv" */
static const unsigned char k_loop_body_conv[] = {
    0x3A, 0x1D, 0x0A, 0x1B, 0x22, 0x04, 'L',  'o',  'o',  'p',  0x2A, 0x13, 0x0A, 0x04, 'b', 'o',
    'd',  'y',  0xA0, 0x01, 0x05, 0x32, 0x08, 0x0A, 0x06, 0x22, 0x04, 'C',  'o',  'n',  'v',
};

/* Same wire layout as k_loop_body_conv but with the inner op_type
 * replaced by "Fake" (4 bytes — same length, no offset shifts). */
static const unsigned char k_loop_body_fake[] = {
    0x3A, 0x1D, 0x0A, 0x1B, 0x22, 0x04, 'L',  'o',  'o',  'p',  0x2A, 0x13, 0x0A, 0x04, 'b', 'o',
    'd',  'y',  0xA0, 0x01, 0x05, 0x32, 0x08, 0x0A, 0x06, 0x22, 0x04, 'F',  'a',  'k',  'e',
};

static char *test_loop_with_allowed_subgraph(void)
{
    char *first_bad = NULL;
    const int err = vmaf_dnn_scan_onnx(k_loop_body_conv, sizeof(k_loop_body_conv), &first_bad);
    mu_assert("Loop with Conv body must be accepted", err == 0);
    mu_assert("first_bad must remain NULL on success", first_bad == NULL);
    return NULL;
}

static char *test_loop_with_forbidden_subgraph(void)
{
    char *first_bad = NULL;
    const int err = vmaf_dnn_scan_onnx(k_loop_body_fake, sizeof(k_loop_body_fake), &first_bad);
    mu_assert("Loop body with forbidden op must be rejected", err == -EPERM);
    mu_assert("first_bad must surface inner op", first_bad && strcmp(first_bad, "Fake") == 0);
    free(first_bad);
    return NULL;
}

static char *test_truncated_varint(void)
{
    /* Single byte with continuation bit set, no follow-up byte. */
    const unsigned char buf[] = {0x80};
    const int err = vmaf_dnn_scan_onnx(buf, sizeof(buf), NULL);
    mu_assert("truncated varint must return -EBADMSG", err == -EBADMSG);
    return NULL;
}

static char *test_overlong_varint(void)
{
    /* 10 bytes all with continuation bit — exceeds the 64-bit varint
     * capacity. pb_read_varint rejects once shift reaches 64. */
    const unsigned char buf[10] = {
        0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80, 0x80,
    };
    const int err = vmaf_dnn_scan_onnx(buf, sizeof(buf), NULL);
    mu_assert("overlong varint must return -EBADMSG", err == -EBADMSG);
    return NULL;
}

static char *test_length_overruns_buffer(void)
{
    /* ModelProto.graph claims 16 bytes follow but buffer holds none. */
    const unsigned char buf[] = {0x3A, 0x10};
    const int err = vmaf_dnn_scan_onnx(buf, sizeof(buf), NULL);
    mu_assert("overrun must return -EBADMSG", err == -EBADMSG);
    return NULL;
}

static char *test_node_length_overruns(void)
{
    /* Graph is self-consistent, but the inner node length overruns its
     * graph-scoped slice. */
    const unsigned char buf[] = {
        0x3A, 0x04, /* graph, len=4              */
        0x0A, 0x10, /* node, len=16 (overrun)    */
        0x22, 0x04  /* padding inside graph      */
    };
    const int err = vmaf_dnn_scan_onnx(buf, sizeof(buf), NULL);
    mu_assert("node overrun must return -EBADMSG", err == -EBADMSG);
    return NULL;
}

static char *test_op_type_overruns(void)
{
    /* NodeProto.op_type declares slen=99 but only 2 bytes follow. */
    const unsigned char buf[] = {
        0x3A, 0x06, 0x0A, 0x04, 0x22, 0x63, 'A', 'B' /* slen=99, only 2 bytes     */
    };
    const int err = vmaf_dnn_scan_onnx(buf, sizeof(buf), NULL);
    mu_assert("op_type overrun must return -EBADMSG", err == -EBADMSG);
    return NULL;
}

static char *test_op_name_too_long(void)
{
    /* Build an op_type string of length 128 (over our 127-cap). */
    enum { SLEN = 128 };
    /* slen=128 varint = 0x80 0x01 (2 bytes) */
    /* NodeProto payload = tag(1) + slen-varint(2) + 128 = 131 bytes */
    /* GraphProto payload = tag(1) + node-len-varint(2) + 131 = 134 bytes (node-len=131: 0x83 0x01) */
    /* ModelProto payload = tag(1) + graph-len-varint(2) + 134 = 137 bytes (graph-len=134: 0x86 0x01) */
    unsigned char buf[137];
    size_t p = 0;
    buf[p++] = 0x3A; /* ModelProto.graph tag       */
    buf[p++] = 0x86;
    buf[p++] = 0x01; /* graph len = 134 varint     */
    buf[p++] = 0x0A; /* GraphProto.node tag        */
    buf[p++] = 0x83;
    buf[p++] = 0x01; /* node len = 131 varint      */
    buf[p++] = 0x22; /* NodeProto.op_type tag      */
    buf[p++] = 0x80;
    buf[p++] = 0x01; /* slen = 128 varint          */
    memset(buf + p, 'A', SLEN);
    p += SLEN;
    mu_assert("buffer fully populated", p == sizeof(buf));

    const int err = vmaf_dnn_scan_onnx(buf, sizeof(buf), NULL);
    mu_assert("overlong op name must return -EPERM", err == -EPERM);
    return NULL;
}

static char *test_zero_length_op_name(void)
{
    /* slen=0 is explicitly rejected (no real op has an empty name). */
    const unsigned char buf[] = {
        0x3A, 0x04, 0x0A, 0x02, 0x22, 0x00 /* op_type, slen=0            */
    };
    const int err = vmaf_dnn_scan_onnx(buf, sizeof(buf), NULL);
    mu_assert("empty op name must return -EPERM", err == -EPERM);
    return NULL;
}

static char *test_skip_unrelated_fields(void)
{
    /* Mix a 32-bit and 64-bit field at the ModelProto level before the
     * graph; the scanner must skip them by wire type. */
    const unsigned char buf[] = {/* field 2 (32-bit): tag = (2<<3)|5 = 0x15, 4 bytes */
                                 0x15, 0xde, 0xad, 0xbe, 0xef,
                                 /* field 3 (64-bit): tag = (3<<3)|1 = 0x19, 8 bytes */
                                 0x19, 0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08,
                                 /* ModelProto.graph */
                                 0x3A, 0x08, 0x0A, 0x06, 0x22, 0x04, 'G', 'e', 'm', 'm'};
    const int err = vmaf_dnn_scan_onnx(buf, sizeof(buf), NULL);
    mu_assert("mixed wire types must be skipped cleanly", err == 0);
    return NULL;
}

static char *test_unsupported_wire_type(void)
{
    /* Group start (wire type 3) is deprecated and rejected. */
    const unsigned char buf[] = {0x0B}; /* field 1, wire 3 */
    const int err = vmaf_dnn_scan_onnx(buf, sizeof(buf), NULL);
    mu_assert("deprecated group wire must return -EBADMSG", err == -EBADMSG);
    return NULL;
}

char *run_tests(void)
{
    mu_run_test(test_null_buffer);
    mu_run_test(test_zero_length);
    mu_run_test(test_no_graph_field);
    mu_run_test(test_allowed_op_conv);
    mu_run_test(test_allowed_multiple_ops);
    mu_run_test(test_loop_top_level_allowed);
    mu_run_test(test_if_after_allowed_now_accepted);
    mu_run_test(test_scan_still_rejected);
    mu_run_test(test_loop_with_allowed_subgraph);
    mu_run_test(test_loop_with_forbidden_subgraph);
    mu_run_test(test_truncated_varint);
    mu_run_test(test_overlong_varint);
    mu_run_test(test_length_overruns_buffer);
    mu_run_test(test_node_length_overruns);
    mu_run_test(test_op_type_overruns);
    mu_run_test(test_op_name_too_long);
    mu_run_test(test_zero_length_op_name);
    mu_run_test(test_skip_unrelated_fields);
    mu_run_test(test_unsupported_wire_type);
    return NULL;
}
