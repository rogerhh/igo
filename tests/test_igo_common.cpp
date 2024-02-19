#include <gtest/gtest.h>

extern "C" {
#include <igo.h>
}

TEST(CommonTests, TestInitTypical) {
    igo_common* igo_cm = (igo_common*) malloc(sizeof(igo_common));
    int ret = igo_init(igo_cm);
    ASSERT_EQ(ret, 1);
    ASSERT_NE(igo_cm, (void*)NULL);
    // check cholmod initialization
    ASSERT_NE(igo_cm->cholmod_cm, (void*)NULL);
    ASSERT_EQ(igo_cm->cholmod_cm->status, 0);
    // check matrices
    ASSERT_NE(igo_cm->A, (void*)NULL);
    ASSERT_NE(igo_cm->b, (void*)NULL);
    ASSERT_NE(igo_cm->L, (void*)NULL);
    ASSERT_NE(igo_cm->PAb, (void*)NULL);
    ASSERT_NE(igo_cm->x, (void*)NULL);
    ASSERT_NE(igo_cm->y, (void*)NULL);
    // cleanup
    cholmod_free_sparse(&igo_cm->A->A, igo_cm->cholmod_cm);
    cholmod_free_dense(&igo_cm->b->B, igo_cm->cholmod_cm);
    cholmod_free_factor(&igo_cm->L->L, igo_cm->cholmod_cm);
    cholmod_free_dense(&igo_cm->PAb->B, igo_cm->cholmod_cm);
    cholmod_free_dense(&igo_cm->x->B, igo_cm->cholmod_cm);
    cholmod_free_dense(&igo_cm->y->B, igo_cm->cholmod_cm);
    free(igo_cm->A);
    free(igo_cm->b);
    free(igo_cm->L);
    free(igo_cm->PAb);
    free(igo_cm->x);
    free(igo_cm->y);
    cholmod_finish(igo_cm->cholmod_cm);
    free(igo_cm);
}

TEST(CommonTests, TestFinishTypical) {
    igo_common* igo_cm = (igo_common*) malloc(sizeof(igo_common));
    igo_init(igo_cm);
    int ret = igo_finish(igo_cm);
    ASSERT_EQ(ret, 1);
    ASSERT_EQ(igo_cm->cholmod_cm, (void*)NULL);
    free(igo_cm);
}

TEST(CommonTests, TestFinishNull) {
    igo_common* igo_cm = NULL;
    int ret = igo_finish(igo_cm);
    ASSERT_EQ(ret, 0);
}

TEST(CommonTests, TestInitNull) {
    igo_common* igo_cm = NULL;
    int ret = igo_init(igo_cm);
    ASSERT_EQ(ret, 0);
}