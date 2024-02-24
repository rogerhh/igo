#include <igo.h>

#define DEFINE_IGO_VECTOR_IMPL(TYPE) \
    igo_vector_##TYPE* igo_allocate_vector_##TYPE (   \
    /* --- input --- */                             \
    int size,                                       \
    /* ------------- */                             \
    igo_common* igo_cm                              \
) {                                                 \
    igo_vector_##TYPE* v = (igo_vector_##TYPE*) malloc(sizeof(igo_vector_##TYPE));  \
    v->maxlen = size > IGO_VECTOR_DEFAULT_SIZE? size : IGO_VECTOR_DEFAULT_SIZE;     \
    v->len = 0;                                                                     \
    v->data = (TYPE*) malloc(v->maxlen * sizeof(TYPE));                             \
    return v;                                                                       \
}                                                                                   \
                                                                                    \
void igo_free_vector_##TYPE (                                                          \
    /* --- input --- */                                                             \
    igo_vector_##TYPE** v_handle,                                                   \
    /* ------------- */                                                             \
    igo_common* igo_cm                                                              \
) {                                                                                 \
    if(!v_handle) { return; }                                                       \
                                                                                    \
    igo_vector_##TYPE* v = *v_handle;                                               \
                                                                                    \
    if(!v) { return; }                                                              \
                                                                                    \
    free(v->data);                                                                  \
    free(v);                                                                        \
                                                                                    \
    *v_handle = NULL;                                                               \
                                                                                    \
    return;                                                                         \
                                                                                    \
}                                                                                   \
                                                                                    \
int igo_resize_vector_##TYPE (                                                      \
    /* --- input --- */                                                             \
    int newsize,                                                                    \
    /* --- in/out --- */                                                            \
    igo_vector_##TYPE* v,                                                           \
    /* ------------- */                                                             \
    igo_common* igo_cm                                                              \
) {                                                                                 \
    v->maxlen = newsize;                                                            \
    v->len = v->maxlen > v->len? v->len : v->maxlen;                                \
    v->data = (TYPE*) realloc(v->data, v->maxlen * sizeof(TYPE));                   \
    return 1;                                                                       \
}                                                                                   \
                                                                                    \
int igo_vector_##TYPE##_multi_pushback (                                            \
    /* --- input --- */                                                             \
    int size,                                                                       \
    TYPE val,                                                                       \
    /* --- in/out --- */                                                            \
    igo_vector_##TYPE* v,                                                           \
    /* ------------- */                                                             \
    igo_common* igo_cm                                                              \
) {                                                                                 \
    if(v->len + size >= v->maxlen) {                                                \
        igo_resize_vector_##TYPE(v->maxlen * 2, v, igo_cm);                         \
    }                                                                               \
                                                                                    \
    int oldlen = v->len;                                                            \
    v->len += size;                                                                 \
                                                                                    \
    for(int i = oldlen; i < v->len; i++) {                                          \
        v->data[i] = val;                                                           \
    }                                                                               \
    return 1;                                                                       \
}                                                                                   \

DEFINE_IGO_VECTOR_IMPL(int);
DEFINE_IGO_VECTOR_IMPL(double);

