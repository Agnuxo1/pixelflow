"""GLSL shader sources and program builder for the moderngl backend."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Vertex shader: full-screen quad, GLSL 3.30 core.
# Emits two triangles covering [-1,1]^2 with UV in [0,1]^2.
# ---------------------------------------------------------------------------

VERTEX_SHADER = """\
#version 330 core

// Vertices are hard-coded; draw 3 vertices for a clip-space triangle that
// covers the entire screen (no buffer upload needed).
void main() {
    // Triangle that covers the NDC square:
    //   vertex 0: (-1, -1)
    //   vertex 1: ( 3, -1)
    //   vertex 2: (-1,  3)
    vec2 pos[3] = vec2[](
        vec2(-1.0, -1.0),
        vec2( 3.0, -1.0),
        vec2(-1.0,  3.0)
    );
    gl_Position = vec4(pos[gl_VertexID], 0.0, 1.0);
}
"""


def build_program(ctx, fragment_src: str):
    """Compile and return a moderngl.Program from the passthrough vertex shader
    and the given fragment shader source.

    Parameters
    ----------
    ctx:
        An active moderngl context.
    fragment_src:
        GLSL 3.30 core fragment shader source.

    Returns
    -------
    moderngl.Program
    """
    return ctx.program(
        vertex_shader=VERTEX_SHADER,
        fragment_shader=fragment_src,
    )
