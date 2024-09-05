struct MainPassPayload {
    vec3 color;
    vec3 normal;
    float t;
};

struct AOPayload {
    float t;
};

#define FLT_MIN 1.175494351e-38
const uint AO_SPP = 8;
const float PI = 3.14159265358979323;