
GrB_Info LAGraph_log 
(
    char *caller,           // calling function
    char *message1,         // message to include (may be NULL)
    char *message2,         // message to include (may be NULL)
    int nthreads,           // # of threads used
    double t                // time taken by the test
) ;


//------------------------------------------------------------------------------
// LAGraph_Context:
//------------------------------------------------------------------------------

// All LAGraph functions will use a Context for global parameters, error
// status, and the like.  So far, the parameter is only for LAGraph_random.

typedef struct
{
    int nthreads ;          // # of threads to use.  If <= 0, use defaults
                            // (from omp_get_max_threads)
}
LAGraph_Context ;

//------------------------------------------------------------------------------
// global objects
//------------------------------------------------------------------------------

// LAGraph_ComplexFP64 operators
//------------------------------------------------------------------------------
// 10 binary functions, z=f(x,y), where CxC -> C
//------------------------------------------------------------------------------

extern
GrB_BinaryOp
    LAGraph_FIRST_ComplexFP64           ,
    LAGraph_SECOND_ComplexFP64          ,
    LAGraph_MIN_ComplexFP64             ,
    LAGraph_MAX_ComplexFP64             ,
    LAGraph_PLUS_ComplexFP64            ,
    LAGraph_MINUS_ComplexFP64           ,
    LAGraph_TIMES_ComplexFP64           ,
    LAGraph_DIV_ComplexFP64             ,
    LAGraph_RDIV_ComplexFP64            ,
    LAGraph_RMINUS_ComplexFP64          ,
    LAGraph_SKEW_ComplexFP64            ,
    LAGraph_PAIR_ComplexFP64            ,
    LAGraph_ANY_ComplexFP64            ,
    LAGraph_HERMITIAN_ComplexFP64       ;

//------------------------------------------------------------------------------
// 6 binary comparison functions, z = f(x,y), where CxC -> C
//------------------------------------------------------------------------------

extern
GrB_BinaryOp
    LAGraph_ISEQ_ComplexFP64              ,
    LAGraph_ISNE_ComplexFP64              ,
    LAGraph_ISGT_ComplexFP64              ,
    LAGraph_ISLT_ComplexFP64              ,
    LAGraph_ISGE_ComplexFP64              ,
    LAGraph_ISLE_ComplexFP64              ;

//------------------------------------------------------------------------------
// 3 binary boolean functions, z=f(x,y), where CxC -> C
//------------------------------------------------------------------------------

extern
GrB_BinaryOp
    LAGraph_OR_ComplexFP64                ,
    LAGraph_AND_ComplexFP64               ,
    LAGraph_XOR_ComplexFP64               ;

//------------------------------------------------------------------------------
// 6 binary comparison functions, z=f(x,y), where CxC -> bool
//------------------------------------------------------------------------------

extern
GrB_BinaryOp
    LAGraph_EQ_ComplexFP64                ,
    LAGraph_NE_ComplexFP64                ,
    LAGraph_GT_ComplexFP64                ,
    LAGraph_LT_ComplexFP64                ,
    LAGraph_GE_ComplexFP64                ,
    LAGraph_LE_ComplexFP64                ;

//------------------------------------------------------------------------------
// 1 binary function, z=f(x,y), where double x double -> C
//------------------------------------------------------------------------------

extern GrB_BinaryOp LAGraph_COMPLEX_ComplexFP64 ;

//------------------------------------------------------------------------------
// 5 unary functions, z=f(x) where C -> C
//------------------------------------------------------------------------------

extern
GrB_UnaryOp
    LAGraph_IDENTITY_ComplexFP64          ,
    LAGraph_AINV_ComplexFP64              ,
    LAGraph_MINV_ComplexFP64              ,
    LAGraph_NOT_ComplexFP64               ,
    LAGraph_CONJ_ComplexFP64              ,
    LAGraph_ONE_ComplexFP64               ,
    LAGraph_ISONE_ComplexFP64             ,
    LAGraph_ABS_ComplexFP64               ,
    LAGraph_TRUE_BOOL_ComplexFP64         ;

//------------------------------------------------------------------------------
// 4 unary functions, z=f(x) where C -> double
//------------------------------------------------------------------------------

extern 
GrB_UnaryOp
    LAGraph_REAL_ComplexFP64              ,
    LAGraph_IMAG_ComplexFP64              ,
    LAGraph_CABS_ComplexFP64              ,
    LAGraph_ANGLE_ComplexFP64             ;

//------------------------------------------------------------------------------
// 2 unary functions, z=f(x) where double -> C
//------------------------------------------------------------------------------

extern GrB_UnaryOp
    LAGraph_COMPLEX_REAL_ComplexFP64      ,
    LAGraph_COMPLEX_IMAG_ComplexFP64      ;

//------------------------------------------------------------------------------
// Complex type, scalars, monoids, and semiring
//------------------------------------------------------------------------------

extern GrB_Type LAGraph_ComplexFP64 ;

extern GrB_Monoid
    LAGraph_PLUS_ComplexFP64_MONOID       ,
    LAGraph_TIMES_ComplexFP64_MONOID      ;
    
extern GrB_Semiring LAGraph_PLUS_TIMES_ComplexFP64 ;

extern double _Complex LAGraph_ComplexFP64_1 ;
extern double _Complex LAGraph_ComplexFP64_0 ;

GrB_Info LAGraph_Complex_init ( ) ;
GrB_Info LAGraph_Complex_finalize ( ) ;

extern GrB_BinaryOp

    // binary operators to test for symmetry, skew-symmetry
    // and Hermitian property
    LAGraph_SKEW_INT8           ,
    LAGraph_SKEW_INT16          ,
    LAGraph_SKEW_INT32          ,
    LAGraph_SKEW_INT64          ,
    LAGraph_SKEW_FP32           ,
    LAGraph_SKEW_FP64           ,
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
    LAGraph_TRUE_BOOL           ;

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

extern GxB_SelectOp LAGraph_support ;

//------------------------------------------------------------------------------
// user-callable utility functions
//------------------------------------------------------------------------------

typedef void (*LAGraph_binary_function) (void *, const void *, const void *) ;

GrB_Info LAGraph_init ( ) ;         // start LAGraph

GrB_Info LAGraph_xinit              // start LAGraph (alternative method)
(
    // pointers to memory management functions
    void * (* user_malloc_function  ) (size_t),
    void * (* user_calloc_function  ) (size_t, size_t),
    void * (* user_realloc_function ) (void *, size_t),
    void   (* user_free_function    ) (void *),
    bool user_malloc_is_thread_safe
) ;

GrB_Info LAGraph_finalize ( ) ;     // end LAGraph

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
                            // LAGraph_ComplexFP64.
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
                            // LAGraph_ComplexFP64.
) ;

GrB_Info LAGraph_Vector_isequal    // return GrB_SUCCESS if successful
(
    bool *result,           // true if A == B, false if A != B or error
    GrB_Vector A,
    GrB_Vector B,
    GrB_BinaryOp userop     // for A and B with arbitrary user-defined types.
                            // Ignored if A and B are of built-in types or
                            // LAGraph_ComplexFP64.
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
    GrB_Type type,              // built-in type, or LAGraph_ComplexFP64
    GrB_Index nrows,            // number of rows
    GrB_Index ncols,            // number of columns
    GrB_Index nvals,            // number of values
    bool make_pattern,          // if true, A is a pattern
    bool make_symmetric,        // if true, A is symmetric
    bool make_skew_symmetric,   // if true, A is skew-symmetric
    bool make_hermitian,        // if true, A is hermitian
    bool no_diagonal,           // if true, A has no entries on the diagonal
    uint64_t *seed              // random number seed; modified on return
) ;

GrB_Info LAGraph_alloc_global ( ) ;

GrB_Info LAGraph_free_global ( ) ;

void *LAGraph_malloc        // wrapper for malloc
(
    size_t nitems,          // number of items
    size_t size_of_item     // size of each item
) ;

void *LAGraph_calloc        // wrapper for calloc
(
    size_t nitems,          // number of items
    size_t size_of_item     // size of each item
) ;

void LAGraph_free           // wrapper for free
(
    void *p
) ;

void LAGraph_tic            // gets current time in seconds and nanoseconds
(
    double tic [2]          // tic [0]: seconds, tic [1]: nanoseconds
) ;

double LAGraph_toc          // returns time since last LAGraph_tic
(
    const double tic [2]    // tic from last call to LAGraph_tic
) ;

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

int LAGraph_set_nthreads        // returns # threads set, 0 if nothing done
(
    int nthreads
) ;

int LAGraph_get_nthreads        // returns # threads to use, 1 if unknown
(
    void
) ;

GrB_Info LAGraph_grread     // read a matrix from a binary file
(
    GrB_Matrix *G,          // handle of matrix to create
    uint64_t *G_version,    // the version in the file
    const char *filename,   // name of file to open
    GrB_Type gtype          // type of matrix to read
) ;

GrB_Info LAGraph_1_to_n     // create an integer vector v = 1:n
(
    GrB_Vector *v_handle,   // vector to create
    GrB_Index n             // size of vector to create
) ;

//------------------------------------------------------------------------------
// user-callable algorithms
//------------------------------------------------------------------------------

GrB_Info LAGraph_bc     // betweeness centrality
(
    GrB_Vector *delta, // delta(i) is the betweeness centrality of node i
    GrB_Matrix A,      // input graph, treated as if boolean in semiring
    GrB_Index s        // source vertex from which to compute shortest paths
);

GrB_Info LAGraph_bc2     // betweeness centrality
(
    GrB_Vector *centrality, // centrality(i): betweeness centrality of node i
    GrB_Matrix A_matrix,    // input graph
    GrB_Index source        // source vertex
) ;

GrB_Info LAGraph_bc_batch // betweeness centrality, batch algorithm
(
    GrB_Vector *delta,  // delta(i) is the betweeness centrality of node i
    const GrB_Matrix A, // input graph, treated as if boolean in semiring
    const GrB_Index *s, // source vertices from which to compute shortest paths
    const int32_t nsver // number of source vertices (length of s)
);

GrB_Info LAGraphX_bc_batch // betweeness centrality, batch algorithm
(
    GrB_Vector *delta,  // delta(i) is the betweeness centrality of node i
    const GrB_Matrix A, // input graph, treated as if boolean in semiring
    const GrB_Index *s, // source vertices from which to compute shortest paths
    const int32_t nsver // number of source vertices (length of s)
);

GrB_Info LAGraphX_bc_batch2 // betweeness centrality, batch algorithm
(
    GrB_Vector *delta,  // delta(i) is the betweeness centrality of node i
    const GrB_Matrix A, // input graph, treated as if boolean in semiring
    const GrB_Index *s, // source vertices from which to compute shortest paths
    const int32_t nsver // number of source vertices (length of s)
);

GrB_Info LAGraphX_bc_batch3 // betweeness centrality, batch algorithm
(
    GrB_Vector *delta,  // delta(i) is the betweeness centrality of node i
    const GrB_Matrix A, // input graph, treated as if boolean in semiring
    const GrB_Matrix AT, // A'
    const GrB_Index *s, // source vertices from which to compute shortest paths
    const int32_t nsver, // number of source vertices (length of s)
    double timing [3]
);

GrB_Info LAGraph_bc_batch3 // betweeness centrality, batch algorithm
(
    GrB_Vector *centrality,    // centrality(i) is the betweeness centrality of node i
    const GrB_Matrix A,        // input graph, treated as if boolean in semiring
    const GrB_Matrix AT,       // A'
    const GrB_Index *sources,  // source vertices from which to compute shortest paths
    int32_t num_sources        // number of source vertices (length of s)
) ;

GrB_Info LAGraph_bc_batch4      // betweeness centrality, batch algorithm
(
    GrB_Vector *centrality,    // centrality(i) is the betweeness centrality of node i
    const GrB_Matrix A_matrix, // input graph, treated as if boolean in semiring
    const GrB_Matrix AT_matrix, // A'
    const GrB_Index *sources,  // source vertices from which to compute shortest paths
    int32_t num_sources        // number of source vertices (length of s)
) ;

GrB_Info LAGraph_bfs_pushpull   // push-pull BFS, or push-only if AT = NULL
(
    GrB_Vector *v_output,   // v(i) is the BFS level of node i in the graph
    GrB_Vector *pi_output,  // pi(i) is the parent of node i in the graph.
                            // if NULL, the parent is not computed
    GrB_Matrix A,           // input graph, treated as if boolean in semiring
    GrB_Matrix AT,          // transpose of A (optional; push-only if NULL)
    int64_t s,              // starting node of the BFS (s < 0: whole graph)
    int64_t max_level,      // optional limit of # levels to search
    bool vsparse            // if true, v is expected to be very sparse
) ;

GrB_Info LAGraph_bfs_parent // push-pull BFS, compute the tree only
(
    // output:
    GrB_Vector *pi_output,  // pi(i) = p+1 if p is the parent of node i
    // inputs:
    GrB_Matrix A,           // input graph, any type
    GrB_Matrix AT,          // transpose of A (optional; push-only if NULL)
    GrB_Vector Degree,      // Degree(i) is the out-degree of node i
                            // (optional: push-only if NULL)
    int64_t source          // starting node of the BFS
) ;

GrB_Info LAGraph_bfs_parent2 // push-pull BFS, compute the tree only
(
    // output:
    GrB_Vector *pi_output,  // pi(i) = p+1 if p is the parent of node i
    // inputs:
    GrB_Matrix A,           // input graph, any type
    GrB_Matrix AT,          // transpose of A (optional; push-only if NULL)
    GrB_Vector Degree,      // Degree(i) is the out-degree of node i
                            // (optional: push-only if NULL)
    int64_t source          // starting node of the BFS
) ;

GrB_Info bfs_log                // push-pull BFS, or push-only if AT = NULL
(
    GrB_Vector *v_output,   // v(i) is the BFS level of node i in the graph
    GrB_Vector *pi_output,  // pi(i) = p if p is the parent of node i.
                            // if NULL, the parent is not computed.
    GrB_Matrix A,           // input graph, treated as if boolean in semiring
    GrB_Matrix AT,          // transpose of A (optional; push-only if NULL)
    int64_t s,              // starting node of the BFS
    int64_t max_level,      // optional limit of # levels to search
    bool vsparse            // if true, v is expected to be very sparse
    , FILE *file
)  ;

GrB_Info LAGraph_bfs_simple     // push-only BFS
(
    GrB_Vector *v_output,   // v(i) is the BFS level of node i in the graph
    const GrB_Matrix A,     // input graph, treated as if boolean in semiring
    GrB_Index s             // starting node of the BFS
) ;

GrB_Info LAGraph_cc_lacc (
    GrB_Vector *result,     // output: array of component identifiers
    GrB_Matrix A,           // input matrix
    bool sanitize           // if true, ensure A is symmetric
) ;

GrB_Info LAGraph_cc_fastsv (
    GrB_Vector *result,     // output: array of component identifiers
    GrB_Matrix A,           // input matrix
    bool sanitize           // if true, ensure A is symmetric
) ;

GrB_Info LAGraph_cc_fastsv2 (
    GrB_Vector *result,     // output: array of component identifiers
    GrB_Matrix A,           // input matrix
    bool sanitize           // if true, ensure A is symmetric
) ;

GrB_Info LAGraph_cc_fastsv3 (
    GrB_Vector *result,     // output: array of component identifiers
    GrB_Matrix A,           // input matrix
    bool sanitize           // if true, ensure A is symmetric
) ;

GrB_Info LAGraph_cc_fastsv4 (
    GrB_Vector *result,     // output: array of component identifiers
    GrB_Matrix A,           // input matrix
    bool sanitize           // if true, ensure A is symmetric
) ;

GrB_Info LAGraph_cc_fastsv5 (
    GrB_Vector *result,     // output: array of component identifiers
    GrB_Matrix A,           // input matrix
    bool sanitize           // if true, ensure A is symmetric
) ;

GrB_Info LAGraph_cc_fastsv5a (
    GrB_Vector *result,     // output: array of component identifiers
    GrB_Matrix *A,          // input matrix
                            //   content remains the same, but pointer changes
    bool sanitize           // if true, ensure A is symmetric
) ;

GrB_Info LAGraph_cc_fastsv5b (
    GrB_Vector *result,     // output: array of component identifiers
    GrB_Matrix *A,          // input matrix
                            //   content remains the same, but pointer changes
    bool sanitize           // if true, ensure A is symmetric
) ;

GrB_Info LAGraph_cc_boruvka (
    GrB_Vector *result,     // output: array of component identifiers
    GrB_Matrix A,           // input matrix
    bool sanitize           // if true, ensure A is symmetric
) ;

GrB_Info LAGraph_msf (
    GrB_Matrix *result,     // output: an unsymmetrical matrix, the spanning forest
    GrB_Matrix A,           // input matrix
    bool sanitize           // if true, ensure A is symmetric
) ;

GrB_Info LAGraph_scc (
    GrB_Vector *result,     // output: array of component identifiers
    GrB_Matrix A            // input matrix
) ;

// LAGraph_pagerank computes an array of structs for its result
typedef struct
{
    double pagerank ;   // the pagerank of a node
    GrB_Index page ;    // the node number itself
}
LAGraph_PageRank ;

GrB_Info LAGraph_pagerank       // GrB_SUCCESS or error condition
(
    LAGraph_PageRank **Phandle, // output: array of LAGraph_PageRank structs
    GrB_Matrix A,               // binary input graph, not modified
    int itermax,                // max number of iterations
    double tol,                 // stop when norm (r-rnew,2) < tol
    int *iters                  // number of iterations taken
) ;

GrB_Info LAGraph_pagerank2      // second PageRank definition
(
    GrB_Vector *result,         // output: array of LAGraph_PageRank structs
    GrB_Matrix A,               // binary input graph, not modified
    double damping_factor,      // damping factor
    unsigned long itermax       // number of iterations
) ;

GrB_Info LAGraph_pagerank3a // PageRank definition
(
    GrB_Vector *result,     // output: array of LAGraph_PageRank structs
    GrB_Matrix A,           // binary input graph, not modified
    GrB_Vector d_out,       // outbound degree of all nodes
    float damping,          // damping factor (typically 0.85)
    int itermax,            // maximum number of iterations
    int *iters              // output: number of iterations taken
) ;

GrB_Info LAGraph_pagerank3b     // PageRank definition
(
    GrB_Vector *result,         // output: array of LAGraph_PageRank structs
    GrB_Matrix A,               // binary input graph, not modified
    float damping_factor,       // damping factor
    unsigned long itermax,      // maximum number of iterations
    int *iters                  // number of iterations taken
) ;

GrB_Info LAGraph_pagerank3c         // PageRank definition
(
    GrB_Vector *result,             // output: array of LAGraph_PageRank structs
    GrB_Matrix A,                   // binary input graph, not modified
    const float * d_out, // out degree of each node (GrB_FP32, size n)
    float damping,                  // damping factor (typically 0.85)
    int itermax,                    // maximum number of iterations
    int* iters                      // output: number of iterations taken
) ;

GrB_Info LAGraph_pagerank3d // PageRank definition
(
    GrB_Vector *result,     // output: array of LAGraph_PageRank structs
    GrB_Matrix A,           // binary input graph, not modified
    GrB_Vector d_out,       // outbound degree of all nodes (not modified)
    float damping,          // damping factor (typically 0.85)
    int itermax,            // maximum number of iterations
    int *iters              // output: number of iterations taken
) ;

GrB_Info LAGraph_pagerank3e // PageRank definition
(
    GrB_Vector *result,     // output: array of LAGraph_PageRank structs
    GrB_Matrix A,           // binary input graph, not modified
    GrB_Vector d_out,       // outbound degree of all nodes (not modified)
    float damping,          // damping factor (typically 0.85)
    int itermax,            // maximum number of iterations
    int *iters              // output: number of iterations taken
) ;

GrB_Info LAGraph_pagerank3f // PageRank definition
(
    GrB_Vector *result,     // output: array of LAGraph_PageRank structs
    GrB_Matrix A,           // binary input graph, not modified
    GrB_Vector d_out,       // outbound degree of all nodes (not modified)
    float damping,          // damping factor (typically 0.85)
    int itermax,            // maximum number of iterations
    int *iters              // output: number of iterations taken
) ;

GrB_Info LAGraph_pagerankx4 // PageRank definition
(
    GrB_Vector *result,     // output: array of LAGraph_PageRank structs
    GrB_Matrix A,           // binary input graph, not modified
    const float * d_out, // out degree of each node (GrB_FP32, size n)
    float damping,          // damping factor (typically 0.85)
    int itermax,            // maximum number of iterations
    int *iters              // output: number of iterations taken
) ;

GrB_Info LAGraph_tricount   // count # of triangles
(
    int64_t *ntri,          // # of triangles
    const int method,       // 1 to 6, see above
    const int sorting,      //  0: no sort
                            //  1: sort by degree, ascending order
                            // -1: sort by degree, descending order
    const int64_t *degree,  // degree of each node, may be NULL if sorting==0.
                            // of size n, unmodified. 
    const GrB_Matrix A_in   // input matrix, must be symmetric, no diag entries
) ;

GrB_Info LAGraph_ktruss         // compute the k-truss of a graph
(
    GrB_Matrix *Chandle,        // output k-truss subgraph, C
    const GrB_Matrix A,         // input adjacency matrix, A, not modified
    const uint32_t k,           // find the k-truss, where k >= 3
    int32_t *nsteps             // # of steps taken (ignored if NULL)
) ;

GrB_Info LAGraph_allktruss      // compute all k-trusses of a graph
(
    GrB_Matrix *Cset,           // size n, output k-truss subgraphs (optional)
    GrB_Matrix A,               // input adjacency matrix, A, not modified
    // output statistics
    int64_t *kmax,              // smallest k where k-truss is empty
    int64_t *ntris,             // size n, ntris [k] is #triangles in k-truss
    int64_t *nedges,            // size n, nedges [k] is #edges in k-truss
    int64_t *nstepss            // size n, nstepss [k] is #steps for k-truss
) ;

GrB_Info LAGraph_BF_full
(
    GrB_Vector *pd,             //the pointer to the vector of distance
    GrB_Vector *ppi,            //the pointer to the vector of parent
    GrB_Vector *ph,             //the pointer to the vector of hops
    const GrB_Matrix A,         //matrix for the graph
    const GrB_Index s           //given index of the source
) ;

GrB_Info LAGraph_BF_basic
(
    GrB_Vector *pd,             //the pointer to the vector of distance
    const GrB_Matrix A,         //matrix for the graph
    const GrB_Index s           //given index of the source
) ;

GrB_Info LAGraph_BF_basic_pushpull
(
    GrB_Vector *pd,             //the pointer to the vector of distance
    const GrB_Matrix A,         //matrix for the graph
    const GrB_Matrix AT,        //transpose of A (optional)
    const GrB_Index s           //given index of the source
) ;

GrB_Info LAGraph_lcc            // compute lcc for all nodes in A
(
    GrB_Vector *LCC_handle,     // output vector
    const GrB_Matrix A,         // input matrix
    bool symmetric,             // if true, the matrix is symmetric
    bool sanitize,              // if true, ensure A is binary
    double t [2]                // t [0] = sanitize time, t [1] = lcc time,
                                // in seconds
) ;

GrB_Info LAGraph_cdlp           // compute cdlp for all nodes in A
(
    GrB_Vector *CDLP_handle,    // output vector
    const GrB_Matrix A,         // input matrix
    bool symmetric,             // denote whether the matrix is symmetric
    bool sanitize,              // if true, ensure A is binary
    int itermax,                // max number of iterations,
    double *t                   // t [0] = sanitize time, t [1] = cdlp time,
                                // in seconds
) ;

GrB_Info LAGraph_dense_relabel   // relabel sparse IDs to dense row/column indices
(
    GrB_Matrix *Id2index_handle, // output matrix: A(id, index)=1 (unfilled if NULL)
    GrB_Matrix *Index2id_handle, // output matrix: B(index, id)=1 (unfilled if NULL)
    GrB_Vector *id2index_handle, // output vector: v(id)=index (unfilled if NULL)
    const GrB_Index *ids,        // array of unique identifiers (under GxB_INDEX_MAX)
    GrB_Index nids,              // number of identifiers
    GrB_Index *id_dimension      // number of rows in Id2index matrix, id2index vector (unfilled if NULL)
) ;

GrB_Info LAGraph_dnn    // returns GrB_SUCCESS if successful
(
    // output
    GrB_Matrix *Yhandle,    // Y, created on output
    // input: not modified
    GrB_Matrix *W,      // W [0..nlayers-1], each nneurons-by-nneurons
    GrB_Matrix *Bias,   // Bias [0..nlayers-1], diagonal nneurons-by-nneurons
    int nlayers,        // # of layers
    GrB_Matrix Y0       // input features: nfeatures-by-nneurons
) ;

GrB_Info LAGraph_sssp // single source shortest paths
(
    GrB_Vector *path_length,   // path_length(i) is the length of the shortest
                               // path from the source vertex to vertex i
    const GrB_Matrix graph,    // input graph, treated as if boolean in semiring
    const GrB_Index source,    // source vertex from which to compute shortest paths
    double delta               // delta value for delta stepping
) ;

GrB_Info LAGraph_sssp1 // single source shortest paths
(
    GrB_Vector *path_length,   // path_length(i) is the length of the shortest
                               // path from the source vertex to vertex i
    GrB_Matrix graph,          // input graph, treated as if boolean in semiring
    GrB_Index source,          // source vertex from which to compute shortest paths
    int32_t delta               // delta value for delta stepping
);


GrB_Info LAGraph_BF_pure_c
(
    int32_t **pd,    // pointer to distance vector d, d(k) = shorstest distance
                     // between s and k if k is reachable from s
    int64_t **ppi,   // pointer to parent index vector pi, pi(k) = parent of
                     // node k in the shortest path tree
    const int64_t s, // given source node index
    const int64_t n, // number of nodes
    const int64_t nz,// number of edges
    const int64_t *I,// row index vector
    const int64_t *J,// column index vector
    const int32_t *W // weight vector, W(i) = weight of edge (I(i),J(i))
);

GrB_Info LAGraph_sssp11         // single source shortest paths
(
    GrB_Vector *path_length,   // path_length(i) is the length of the shortest
                               // path from the source vertex to vertex i
    GrB_Matrix A,              // input graph, treated as if boolean in
                               // semiring (INT32)
    GrB_Index source,          // source vertex from which to compute
                               // shortest paths
    int32_t delta,             // delta value for delta stepping
    bool AIsAllPositive        // A boolean indicating whether the entries of
                               // matrix A are all positive
);

GrB_Info LAGraph_sssp12         // single source shortest paths
(
    GrB_Vector *path_length,   // path_length(i) is the length of the shortest
                               // path from the source vertex to vertex i
    GrB_Matrix A,              // input graph, treated as if boolean in
                               // semiring (INT32)
    GrB_Index source,          // source vertex from which to compute
                               // shortest paths
    int32_t delta,             // delta value for delta stepping
    bool AIsAllPositive        // A boolean indicating whether the entries of
                               // matrix A are all positive
);


GrB_Info LAGraph_sssp12c        // single source shortest paths
(
    GrB_Vector *path_length,   // path_length(i) is the length of the shortest
                               // path from the source vertex to vertex i
    GrB_Matrix A,              // input graph, treated as if boolean in
                               // semiring (INT32)
    GrB_Index source,          // source vertex from which to compute
                               // shortest paths
    int32_t delta,             // delta value for delta stepping

    // TODO: make this an enum:
    //      case 0: A can have negative, zero, or positive entries
    //      case 1: A can have zero or positive entries
    //      case 2: A only has positive entries (see FIXME below)
    bool AIsAllPositive        // A boolean indicating whether the entries of
                               // matrix A are all positive
);


GrB_Info LAGraph_bfs_both       // push-pull BFS, or push-only if AT = NULL
(
    GrB_Vector *v_output,   // v(i) is the BFS level of node i in the graph
    GrB_Vector *pi_output,  // pi(i) = p+1 if p is the parent of node i.
                            // if NULL, the parent is not computed.
    GrB_Matrix A,           // input graph, treated as if boolean in semiring
    GrB_Matrix AT,          // transpose of A (optional; push-only if NULL)
    int64_t source,         // starting node of the BFS
    int64_t max_level,      // optional limit of # levels to search
    bool vsparse            // if true, v is expected to be very sparse
    , FILE * logfile
) ;


GrB_Info LAGraph_Matrix_extract_keep_dimensions // extract submatrix but keep
                                                // the dimensions of the
                                                // original matrix
(
    GrB_Matrix *Chandle,         // output matrix
    const GrB_Matrix A,          // input matrix
    const GrB_Index *Vsparse,    // sorted list of vertex indices
    const bool *Vdense,          // boolean array of vertices
    GrB_Index nv                 // number of vertex indices
) ;
