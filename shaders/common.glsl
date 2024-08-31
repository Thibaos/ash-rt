#define KIND_TOP_FACE 0
#define KIND_LEFT_FACE 1
#define KIND_BACK_FACE 2
#define KIND_RIGHT_FACE 3
#define KIND_FRONT_FACE 4
#define KIND_BOTTOM_FACE 5
#define KIND_UNKNOWN 6

struct RayPayload {
    vec3 color;
    vec3 origin;
    vec3 direction;
    float attenuation;
    float t;
};