#version 460

// RTAO - Low-pass filtering shader, Jakub Boksansky 2018

// ========================================================================
// Constant buffer inputs
// ========================================================================

struct LowPassFilterInfo
{
    mat4 normalMatrix;
	vec2 texelSize;
    int outputMode;
};

layout(location = 0) in vec3 vertex_tex_coords;
layout(location = 0) out vec4 frag_color;

layout(set = 0, binding = 0) uniform _LowPassFilterInfo { LowPassFilterInfo filterInfo; };

// ========================================================================
// Textures and Samplers
// ========================================================================

layout(set = 0, binding = 1) uniform sampler2D depth_normals_texture;
layout(set = 0, binding = 2) uniform sampler2D ao_texture;
layout(set = 0, binding = 3) uniform sampler2D color_texture;

// ========================================================================
// Pixel Shaders 
// ========================================================================

const float gauss[5] = float[5](0.05448868, 0.2442013, 0.40262, 0.2442013, 0.05448868);

bool isValidTap(float tapDepth, float centerDepth, vec3 tapNormal, vec3 centerNormal, float dotViewNormal) {
	const float depthRelativeDifferenceEpsilonMin = 0.003;
	const float depthRelativeDifferenceEpsilonMax = 0.02;
	const float dotNormalsEpsilon = 0.9;

	// Adjust depth difference epsilon based on view space normal
    float depthRelativeDifferenceEpsilon = mix(depthRelativeDifferenceEpsilonMax, depthRelativeDifferenceEpsilonMin, dotViewNormal);

	// Check depth
	if (abs(1.0 - (tapDepth / centerDepth)) > depthRelativeDifferenceEpsilon) return false;

	// Check normals
	if (dot(tapNormal, centerNormal) < dotNormalsEpsilon) return false;

	return true;
}

// Filter ambient occlusion using low-pass tent filter and mix it with color buffer
float lowPassFilter(vec2 filterDirection, const bool doTemporalFilter) {
    float ao = 0.0;
	float weight = 1.0;

	vec4 centerDepthNormal = texture(depth_normals_texture, gl_FragCoord.xy);
	float centerDepth = centerDepthNormal.a;
	vec3 centerNormal = normalize(centerDepthNormal.rgb);
    float dotViewNormal = abs((filterInfo.normalMatrix * vec4(centerNormal, 0.0)).z);

    vec2 offsetScale = filterInfo.texelSize * filterDirection;

    for (int i = -2; i <= 2; ++i) {
        vec2 offset = float(i) * offsetScale;

		vec4 tapDepthNormal = texture(depth_normals_texture , gl_FragCoord.xy + offset);
		vec4 tapAO = texture(ao_texture, gl_FragCoord.xy + offset);
		float tapDepth = tapDepthNormal.a;
		vec3 tapNormal = normalize(tapDepthNormal.rgb);

		float tapWeight = gauss[i + 2];

        if (isValidTap(tapDepth, centerDepth, tapNormal, centerNormal, dotViewNormal)) {
            ao += (doTemporalFilter ? dot(tapAO, vec4(0.25)) : tapAO.r) * tapWeight;
		} else {
			weight -= tapWeight;
		}
    }

	ao /= weight;

	return ao;
}

vec4 low_pass_filter_x()
{
    return lowPassFilter(vec2(1.0, 0.0), true).xxxx;
}

vec4 low_pass_filter_y()
{
    float ao = lowPassFilter(vec2(0.0, 1.0), false);

	vec4 color = texture(color_texture, gl_FragCoord.xy);

    if (filterInfo.outputMode == 0) return vec4(ao);
    if (filterInfo.outputMode == 2) return color;

	return vec4(color.rgb * ao, 1.0);
}

void main() {
	// vec4 x_pass = low_pass_filter_x();
	// vec4 y_pass = low_pass_filter_y();

	// frag_color = mix(x_pass, y_pass, 0.5);

	frag_color = vec4(vertex_tex_coords, 1.0);
}