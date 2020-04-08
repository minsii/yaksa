/*
 * Copyright (C) by Argonne National Laboratory
 *     See COPYRIGHT in top-level directory
 */

#include "yaksi.h"
#include "yaksu.h"
#include "yaksuri_seqi.h"

int yaksuri_seq_init_hook(void)
{
    int rc = YAKSA_SUCCESS;

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksuri_seq_finalize_hook(void)
{
    int rc = YAKSA_SUCCESS;

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksuri_seq_type_create_hook(yaksi_type_s * type, yaksur_seq_pup_fn * pack,
                                 yaksur_seq_pup_fn * unpack)
{
    int rc = YAKSA_SUCCESS;

    rc = yaksuri_seqi_populate_pupfns(type, pack, unpack);
    YAKSU_ERR_CHECK(rc, fn_fail);

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}

int yaksuri_seq_type_free_hook(yaksi_type_s * type)
{
    int rc = YAKSA_SUCCESS;

  fn_exit:
    return rc;
  fn_fail:
    goto fn_exit;
}
