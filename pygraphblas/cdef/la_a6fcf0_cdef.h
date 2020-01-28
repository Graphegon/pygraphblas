//------------------------------------------------------------------------------
// LAGraph.h:  include file for user applications that use LAGraph
//------------------------------------------------------------------------------

/*
    LAGraph:  graph algorithms based on GraphBLAS

    Copyright 2019 LAGraph Contributors.

    (see Contributors.txt for a full list of Contributors; see
    ContributionInstructions.txt for information on how you can Contribute to
    this project).

    All Rights Reserved.

    NO WARRANTY. THIS MATERIAL IS FURNISHED ON AN "AS-IS" BASIS. THE LAGRAPH
    CONTRIBUTORS MAKE NO WARRANTIES OF ANY KIND, EITHER EXPRESSED OR IMPLIED,
    AS TO ANY MATTER INCLUDING, BUT NOT LIMITED TO, WARRANTY OF FITNESS FOR
    PURPOSE OR MERCHANTABILITY, EXCLUSIVITY, OR RESULTS OBTAINED FROM USE OF
    THE MATERIAL. THE CONTRIBUTORS DO NOT MAKE ANY WARRANTY OF ANY KIND WITH
    RESPECT TO FREEDOM FROM PATENT, TRADEMARK, OR COPYRIGHT INFRINGEMENT.

    Released under a BSD license, please see the LICENSE file distributed with
    this Software or contact permission@sei.cmu.edu for full terms.

    Created, in part, with funding and support from the United States
    Government.  (see Acknowledgments.txt file).

    This program includes and/or can make use of certain third party source
    code, object code, documentation and other files ("Third Party Software").
    See LICENSE file for more details.

*/

//------------------------------------------------------------------------------

// TODO: add more comments to this file.

//------------------------------------------------------------------------------
// include files and global #defines
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
// memory management functions
//------------------------------------------------------------------------------

// use the ANSI C functions by default (or mx* functions if the #ifdef
// above redefines them).  See Source/Utility/LAGraph_malloc.c.

extern GrB_BinaryOp

    // binary operators to test for symmetry, skew-symmetry
    // and Hermitian property
    LAGraph_EQ_Complex          ,
    LAGraph_SKEW_INT8           ,
    LAGraph_SKEW_INT16          ,
    LAGraph_SKEW_INT32          ,
    LAGraph_SKEW_INT64          ,
    LAGraph_SKEW_FP32           ,
    LAGraph_SKEW_FP64           ,
    LAGraph_SKEW_Complex        ,
    LAGraph_Hermitian           ,
    LAGraph_LOR_UINT32          ,
    LAGraph_LOR_INT64           ;

extern GrB_UnaryOp

    // unary operators to check if the entry is equal to 1
    LAGraph_ISONE_INT8          ,
    LAGraph_ISONE_INT16         ,
    LAGraph_ISONE_INT32         ,
    LAGraph_ISONE_INT64         ,
    LAGraph_ISONE_UINT8         ,
    LAGraph_ISONE_UINT16        ,
    LAGraph_ISONE_UINT32        ,
    LAGraph_ISONE_UINT64        ,
    LAGraph_ISONE_FP32          ,
    LAGraph_ISONE_FP64          ,
    LAGraph_ISONE_Complex       ,

    // unary operators to check if the entry is equal to 2
    LAGraph_ISTWO_UINT32        ,
    LAGraph_ISTWO_INT64         ,

    // unary operators that decrement by 1
    LAGraph_DECR_INT32          ,
    LAGraph_DECR_INT64          ,

    // unary operators for lcc
    LAGraph_COMB_DIR_FP64       ,
    LAGraph_COMB_UNDIR_FP64     ,

    // unary ops to check if greater than zero
    LAGraph_GT0_FP32            ,
    LAGraph_GT0_FP64            ,

    // unary YMAX ops for DNN
    LAGraph_YMAX_FP32           ,
    LAGraph_YMAX_FP64           ,

    // unary operators that return 1
    LAGraph_ONE_UINT32          ,
    LAGraph_ONE_INT64           ,
    LAGraph_ONE_FP64            ,
    LAGraph_TRUE_BOOL           ,
    LAGraph_TRUE_BOOL_Complex   ;

// monoids and semirings
extern GrB_Monoid

    LAGraph_PLUS_INT64_MONOID   ,
    LAGraph_MAX_INT32_MONOID    ,
    LAGraph_LAND_MONOID         ,
    LAGraph_LOR_MONOID          ,
    LAGraph_MIN_INT32_MONOID    ,
    LAGraph_MIN_INT64_MONOID    ,
    LAGraph_PLUS_UINT32_MONOID  ,
    LAGraph_PLUS_FP32_MONOID    ,
    LAGraph_PLUS_FP64_MONOID    ,
    LAGraph_DIV_FP64_MONOID     ;

extern GrB_Semiring

    LAGraph_LOR_LAND_BOOL       ,
    LAGraph_LOR_SECOND_BOOL     ,
    LAGraph_LOR_FIRST_BOOL      ,
    LAGraph_MIN_SECOND_INT32    ,
    LAGraph_MIN_FIRST_INT32     ,
    LAGraph_MIN_SECOND_INT64    ,
    LAGraph_MIN_FIRST_INT64     ,
    LAGraph_PLUS_TIMES_UINT32   ,
    LAGraph_PLUS_TIMES_INT64    ,
    LAGraph_PLUS_TIMES_FP64     ,
    LAGraph_PLUS_PLUS_FP64      ,
    LAGraph_PLUS_TIMES_FP32     ,
    LAGraph_PLUS_PLUS_FP32      ;

// all 16 descriptors
// syntax: 4 characters define the following.  'o' is the default:
// 1: o or t: A transpose
// 2: o or t: B transpose
// 3: o or c: complemented mask
// 4: o or r: replace
extern GrB_Descriptor

    LAGraph_desc_oooo ,   // default (NULL)
    LAGraph_desc_ooor ,   // replace
    LAGraph_desc_ooco ,   // compl mask
    LAGraph_desc_oocr ,   // compl mask, replace

    LAGraph_desc_tooo ,   // A'
    LAGraph_desc_toor ,   // A', replace
    LAGraph_desc_toco ,   // A', compl mask
    LAGraph_desc_tocr ,   // A', compl mask, replace

    LAGraph_desc_otoo ,   // B'
    LAGraph_desc_otor ,   // B', replace
    LAGraph_desc_otco ,   // B', compl mask
    LAGraph_desc_otcr ,   // B', compl mask, replace

    LAGraph_desc_ttoo ,   // A', B'
    LAGraph_desc_ttor ,   // A', B', replace
    LAGraph_desc_ttco ,   // A', B', compl mask
    LAGraph_desc_ttcr ;   // A', B', compl mask, replace

typedef void (*LAGraph_binary_function) (void *, const void *, const void *) ;

GrB_Info LAGraph_init (void) ;         // start LAGraph

GrB_Info LAGraph_xinit              // start LAGraph (alternative method)
(
    // pointers to memory management functions
    void * (* user_malloc_function  ) (size_t),
    void * (* user_calloc_function  ) (size_t, size_t),
    void * (* user_realloc_function ) (void *, size_t),
    void   (* user_free_function    ) (void *),
    bool user_malloc_is_thread_safe
) ;

GrB_Info LAGraph_finalize (void) ;     // end LAGraph

GrB_Info LAGraph_mmread
(
    GrB_Matrix *A,      // handle of matrix to create
    FILE *f             // file to read from, already open
) ;

GrB_Info LAGraph_mmwrite
(
    GrB_Matrix A,           // matrix to write to the file
    FILE *f                 // file to write it to
    // TODO , FILE *fcomments         // optional file with extra comments
) ;

// ascii header prepended to all *.grb files
#define LAGRAPH_BIN_HEADER ...

GrB_Info LAGraph_binwrite
(
    GrB_Matrix *A,          // matrix to write to the file
    char *filename,         // file to write it to
    const char *comments    // comments to add to the file, up to 220 characters
                            // in length, not including the terminating null
                            // byte. Ignored if NULL.  Characters past
                            // the 220 limit are silently ignored.
) ;

GrB_Info LAGraph_binread
(
    GrB_Matrix *A,          // matrix to read from the file
    char *filename          // file to read it from
) ;

GrB_Info LAGraph_tsvread        // returns GrB_SUCCESS if successful
(
    GrB_Matrix *Chandle,        // C, created on output
    FILE *f,                    // file to read from (already open)
    GrB_Type type,              // the type of C to create
    GrB_Index nrows,            // C is nrows-by-ncols
    GrB_Index ncols
) ;

GrB_Info LAGraph_ispattern  // return GrB_SUCCESS if successful
(
    bool *result,           // true if A is all one, false otherwise
    GrB_Matrix A,
    GrB_UnaryOp userop      // for A with arbitrary user-defined type.
                            // Ignored if A and B are of built-in types or
                            // LAGraph_Complex.
) ;

GrB_Info LAGraph_pattern    // return GrB_SUCCESS if successful
(
    GrB_Matrix *C,          // a boolean matrix with the pattern of A
    GrB_Matrix A,
    GrB_Type T              // return type for C
) ;

GrB_Info LAGraph_isequal    // return GrB_SUCCESS if successful
(
    bool *result,           // true if A == B, false if A != B or error
    GrB_Matrix A,
    GrB_Matrix B,
    GrB_BinaryOp userop     // for A and B with arbitrary user-defined types.
                            // Ignored if A and B are of built-in types or
                            // LAGraph_Complex.
) ;

GrB_Info LAGraph_Vector_isequal    // return GrB_SUCCESS if successful
(
    bool *result,           // true if A == B, false if A != B or error
    GrB_Vector A,
    GrB_Vector B,
    GrB_BinaryOp userop     // for A and B with arbitrary user-defined types.
                            // Ignored if A and B are of built-in types or
                            // LAGraph_Complex.
) ;

GrB_Info LAGraph_isall      // return GrB_SUCCESS if successful
(
    bool *result,           // true if A == B, false if A != B or error
    GrB_Matrix A,
    GrB_Matrix B,
    GrB_BinaryOp op         // GrB_EQ_<type>, for the type of A and B,
                            // to check for equality.  Or use any desired
                            // operator.  The operator should return GrB_BOOL.
) ;

GrB_Info LAGraph_Vector_isall      // return GrB_SUCCESS if successful
(
    bool *result,           // true if A == B, false if A != B or error
    GrB_Vector A,
    GrB_Vector B,
    GrB_BinaryOp op         // GrB_EQ_<type>, for the type of A and B,
                            // to check for equality.  Or use any desired
                            // operator.  The operator should return GrB_BOOL.
) ;

uint64_t LAGraph_rand (uint64_t *seed) ;

uint64_t LAGraph_rand64 (uint64_t *seed) ;

double LAGraph_randx (uint64_t *seed) ;

GrB_Info LAGraph_random         // create a random matrix
(
    GrB_Matrix *A,              // handle of matrix to create
    GrB_Type type,              // built-in type, or LAGraph_Complex
    GrB_Index nrows,            // number of rows
    GrB_Index ncols,            // number of columns
    GrB_Index nvals,            // number of values
    bool make_pattern,          // if true, A is a pattern
    bool make_symmetric,        // if true, A is symmetric
    bool make_skew_symmetric,   // if true, A is skew-symmetric
    bool make_hermitian,        // if trur, A is hermitian
    bool no_diagonal,           // if true, A has no entries on the diagonal
    uint64_t *seed              // random number seed; modified on return
) ;

GrB_Info LAGraph_alloc_global (void) ;

GrB_Info LAGraph_free_global (void) ;

GrB_Info LAGraph_prune_diag // remove all entries from the diagonal
(
    GrB_Matrix A
) ;

GrB_Info LAGraph_Vector_to_dense
(
    GrB_Vector *vdense,     // output vector
    GrB_Vector v,           // input vector
    void *id                // pointer to value to fill vdense with
) ;

GrB_Info LAGraph_1_to_n     // create an integer vector v = 1:n
(
    GrB_Vector *v_handle,   // vector to create
    GrB_Index n             // size of vector to create
) ;
