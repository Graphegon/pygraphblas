//------------------------------------------------------------------------------
// LAGraph_binwrite:  write a matrix to a binary file
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

// LAGraph_binwrite:  write a matrix to a binary file
// Contributed by Tim Davis, Texas A&M.

// Writes a matrix to a file in a binary format.

#include "LAGraph_internal.h"

#define LAGRAPH_FREE_ALL    \
{                           \
    GrB_free (A) ;          \
    LAGRAPH_FREE (Ap) ;     \
    LAGRAPH_FREE (Ah) ;     \
    LAGRAPH_FREE (Ai) ;     \
    LAGRAPH_FREE (Ax) ;     \
}

#define FWRITE(p,s,n)                                               \
{                                                                   \
    size_t result = fwrite (p, s, n, f) ;                           \
    if (result != n)                                                \
    {                                                               \
        fclose (f) ;                                                \
        LAGRAPH_ERROR ("File I/O error", GrB_INVALID_VALUE) ;       \
    }                                                               \
}

//------------------------------------------------------------------------------
// LAGraph_binwrite
//------------------------------------------------------------------------------

GrB_Info LAGraph_binwrite
(
    GrB_Matrix *A,          // matrix to write to the file
    char *filename,         // file to write it to
    const char *comments    // comments to add to the file, up to 220 characters
                            // in length, not including the terminating null
                            // byte. Ignored if NULL.  Characters past
                            // the 220 limit are silently ignored.
)
{

    GrB_Index *Ap = NULL, *Ai = NULL, *Ah = NULL ;
    void *Ax = NULL ;
    int8_t *Ab = NULL ;

    //--------------------------------------------------------------------------
    // check inputs
    //--------------------------------------------------------------------------

    GrB_Info info ;
    if (A == NULL || *A == NULL || filename == NULL)
    {
        // input arguments invalid
        LAGRAPH_ERROR ("LAGraph_binwrite: invalid inputs\n", GrB_NULL_POINTER) ;
    }

    FILE *f = fopen (filename, "w") ;
    if (f == NULL)
    {
        LAGRAPH_ERROR ("LAGraph_binwrite: file cannot be opened",
            GrB_INVALID_VALUE) ;
    }

    GrB_Index ignore ;
    LAGRAPH_OK (GrB_Matrix_nvals (&ignore, *A)) ;
    LAGRAPH_OK (GrB_Matrix_wait (A)) ;

    //--------------------------------------------------------------------------
    // determine the basic matrix properties
    //--------------------------------------------------------------------------

    GxB_Format_Value fmt = -999 ;
    LAGRAPH_OK (GxB_get (*A, GxB_FORMAT, &fmt)) ;

    bool is_hyper = false ;
    bool is_sparse = false ;
    bool is_bitmap = false ;
    bool is_full  = false ;
    LAGRAPH_OK (GxB_get (*A, GxB_IS_HYPER, &is_hyper)) ;
    int32_t kind ;
    double hyper = -999 ;

    #if GxB_IMPLEMENTATION >= GxB_VERSION (4,0,1)
    LAGRAPH_OK (GxB_get (*A, GxB_HYPER_SWITCH, &hyper)) ;
    LAGRAPH_OK (GxB_get (*A, GxB_SPARSITY_STATUS, &kind)) ;
    #else
    kind = is_hyper ? 1 : 0 ;
    LAGRAPH_OK (GxB_get (*A, GxB_HYPER, &hyper)) ;
    #endif

    switch (kind)
    {
        default : 
        case 0 : // for backward compatibility with prior versions
        case 2 : is_sparse = true ; break ; // GxB_SPARSE = 2
        case 1 : is_hyper  = true ; break ; // GxB_HYPERSPARSE = 1
        case 4 : is_bitmap = true ; break ; // GxB_BITMAP = 4
        case 8 : is_full   = true ; break ; // GxB_FULL = 4
    }

    //--------------------------------------------------------------------------
    // export the matrix
    //--------------------------------------------------------------------------

    GrB_Type type ;
    GrB_Index nrows, ncols, nvals, nvec ;
    LAGRAPH_OK (GrB_Matrix_nvals (&nvals, *A)) ;
    size_t typesize ;
    int64_t nonempty = -1 ;
    char *fmt_string ;
    bool jumbled ;
    GrB_Index Ap_size, Ah_size, Ab_size, Ai_size, Ax_size ;

    if (fmt == GxB_BY_COL && is_hyper)
    {
        // hypersparse CSC
        #if GxB_IMPLEMENTATION >= GxB_VERSION (4,0,1)
        LAGRAPH_OK (GxB_Matrix_export_HyperCSC (A, &type, &nrows, &ncols,
            &Ap, &Ah, &Ai, &Ax, &Ap_size, &Ah_size, &Ai_size, &Ax_size,
            &nvec, &jumbled, NULL)) ;
        #elif GxB_IMPLEMENTATION == GxB_VERSION (4,0,0)
        LAGRAPH_OK (GxB_Matrix_export_HyperCSC (A, &type, &nrows, &ncols,
            &Ai_size, &jumbled, &nonempty, &nvec, &Ah, &Ap, &Ai, &Ax, NULL)) ;
        Ap_size = nvec+1 ;
        Ah_size = nvec ;
        Ax_size = Ai_size ;
        #else
        LAGRAPH_OK (GxB_Matrix_export_HyperCSC (A, &type, &nrows, &ncols,
            &Ai_size, &nonempty, &Ah_size, &Ah, &Ap, &Ai, &Ax, NULL)) ;
        Ap_size = nvec+1 ;
        Ah_size = nvec ;
        Ax_size = Ai_size ;
        #endif
        fmt_string = "HCSC" ;
    }
    else if (fmt == GxB_BY_ROW && is_hyper)
    {
        // hypersparse CSR
        #if GxB_IMPLEMENTATION >= GxB_VERSION (4,0,1)
        LAGRAPH_OK (GxB_Matrix_export_HyperCSR (A, &type, &nrows, &ncols,
            &Ap, &Ah, &Ai, &Ax, &Ap_size, &Ah_size, &Ai_size, &Ax_size,
            &nvec, &jumbled, NULL)) ;
        #elif GxB_IMPLEMENTATION == GxB_VERSION (4,0,0)
        LAGRAPH_OK (GxB_Matrix_export_HyperCSR (A, &type, &nrows, &ncols,
            &Ai_size, &jumbled, &nonempty, &nvec, &Ah, &Ap, &Ai, &Ax, NULL)) ;
        Ap_size = nvec+1 ;
        Ah_size = nvec ;
        Ax_size = Ai_size ;
        #else
        LAGRAPH_OK (GxB_Matrix_export_HyperCSR (A, &type, &nrows, &ncols,
            &Ai_size, &nonempty, &Ah_size, &Ah, &Ap, &Ai, &Ax, NULL)) ;
        Ap_size = nvec+1 ;
        Ah_size = nvec ;
        Ax_size = Ai_size ;
        #endif
        fmt_string = "HCSR" ;
    }
    else if (fmt == GxB_BY_COL && is_sparse)
    {
        // standard CSC
        #if GxB_IMPLEMENTATION >= GxB_VERSION (4,0,1)
        LAGRAPH_OK (GxB_Matrix_export_CSC (A, &type, &nrows, &ncols,
            &Ap, &Ai, &Ax, &Ap_size, &Ai_size, &Ax_size,
            &jumbled, NULL)) ;
        #elif GxB_IMPLEMENTATION == GxB_VERSION (4,0,0)
        LAGRAPH_OK (GxB_Matrix_export_CSC (A, &type, &nrows, &ncols,
            &Ai_size, &jumbled, &nonempty, &Ap, &Ai, &Ax, NULL)) ;
        Ap_size = ncols+1 ;
        Ax_size = Ai_size ;
        #else
        LAGRAPH_OK (GxB_Matrix_export_CSC (A, &type, &nrows, &ncols, &Ai_size,
            &nonempty, &Ap, &Ai, &Ax, NULL)) ;
        Ap_size = ncols+1 ;
        Ax_size = Ai_size ;
        #endif
        nvec = ncols ;
        fmt_string = "CSC " ;
    }
    else if (fmt == GxB_BY_ROW && is_sparse)
    {
        // standard CSR
        #if GxB_IMPLEMENTATION >= GxB_VERSION (4,0,1)
        LAGRAPH_OK (GxB_Matrix_export_CSR (A, &type, &nrows, &ncols,
            &Ap, &Ai, &Ax, &Ap_size, &Ai_size, &Ax_size, &jumbled, NULL)) ;
        #elif GxB_IMPLEMENTATION == GxB_VERSION (4,0,0)
        LAGRAPH_OK (GxB_Matrix_export_CSR (A, &type, &nrows, &ncols,
            &Ai_size, &jumbled, &nonempty, &Ap, &Ai, &Ax, NULL)) ;
        Ap_size = ncols+1 ;
        Ax_size = Ai_size ;
        #else
        LAGRAPH_OK (GxB_Matrix_export_CSR (A, &type, &nrows, &ncols,
            &Ai_size, &nonempty, &Ap, &Ai, &Ax, NULL)) ;
        Ap_size = ncols+1 ;
        Ax_size = Ai_size ;
        #endif
        nvec = nrows ;
        fmt_string = "CSR " ;
    }
    else if (fmt == GxB_BY_COL && is_bitmap)
    {
        // bitmap by col
        #if GxB_IMPLEMENTATION >= GxB_VERSION (4,0,1)
        LAGRAPH_OK (GxB_Matrix_export_BitmapC (A, &type, &nrows, &ncols,
            &Ab, &Ax, &Ab_size, &Ax_size, &nvals, NULL)) ;
        #endif
        nvec = ncols ;
        fmt_string = "BITMAPC" ;
    }
    else if (fmt == GxB_BY_ROW && is_bitmap)
    {
        // bitmap by row
        #if GxB_IMPLEMENTATION >= GxB_VERSION (4,0,1)
        LAGRAPH_OK (GxB_Matrix_export_BitmapR (A, &type, &nrows, &ncols,
            &Ab, &Ax, &Ab_size, &Ax_size, &nvals, NULL)) ;
        #endif
        nvec = nrows ;
        fmt_string = "BITMAPR" ;
    }
    else if (fmt == GxB_BY_COL && is_full)
    {
        // full by col
        #if GxB_IMPLEMENTATION >= GxB_VERSION (4,0,1)
        LAGRAPH_OK (GxB_Matrix_export_FullC (A, &type, &nrows, &ncols,
            &Ax, &Ax_size, NULL)) ;
        #endif
        nvec = ncols ;
        fmt_string = "FULLC" ;
    }
    else if (fmt == GxB_BY_ROW && is_full)
    {
        // full by row
        #if GxB_IMPLEMENTATION >= GxB_VERSION (4,0,1)
        LAGRAPH_OK (GxB_Matrix_export_FullR (A, &type, &nrows, &ncols,
            &Ax, &Ax_size, NULL)) ;
        #endif
        nvec = nrows ;
        fmt_string = "FULLC" ;
    }
    else
    {
        LAGRAPH_ERROR ("unknown", GrB_INVALID_VALUE) ;
    }

    //--------------------------------------------------------------------------
    // create the type string
    //--------------------------------------------------------------------------

    LAGRAPH_OK (GxB_Type_size (&typesize, type)) ;

    #define LEN LAGRAPH_BIN_HEADER
    char typename [LEN] ;
    int32_t typecode ;
    if      (type == GrB_BOOL  )
    {
        snprintf (typename, LEN, "GrB_BOOL  ") ;
        typecode = 0 ;
    }
    else if (type == GrB_INT8  )
    {
        snprintf (typename, LEN, "GrB_INT8  ") ;
        typecode = 1 ;
    }
    else if (type == GrB_INT16 )
    {
        snprintf (typename, LEN, "GrB_INT16 ") ;
        typecode = 2 ;
    }
    else if (type == GrB_INT32 )
    {
        snprintf (typename, LEN, "GrB_INT32 ") ;
        typecode = 3 ;
    }
    else if (type == GrB_INT64 )
    {
        snprintf (typename, LEN, "GrB_INT64 ") ;
        typecode = 4 ;
    }
    else if (type == GrB_UINT8 )
    {
        snprintf (typename, LEN, "GrB_UINT8 ") ;
        typecode = 5 ;
    }
    else if (type == GrB_UINT16)
    {
        snprintf (typename, LEN, "GrB_UINT16") ;
        typecode = 6 ;
    }
    else if (type == GrB_UINT32)
    {
        snprintf (typename, LEN, "GrB_UINT32") ;
        typecode = 7 ;
    }
    else if (type == GrB_UINT64)
    {
        snprintf (typename, LEN, "GrB_UINT64") ;
        typecode = 8 ;
    }
    else if (type == GrB_FP32  )
    {
        snprintf (typename, LEN, "GrB_FP32  ") ;
        typecode = 9 ;
    }
    else if (type == GrB_FP64  )
    {
        snprintf (typename, LEN, "GrB_FP64  ") ;
        typecode = 10 ;
    }
    else if (type == LAGraph_ComplexFP64)
    {
        snprintf (typename, LEN, "USER      ") ;
        typecode = 11 ;
    }
    else
    {
        // TODO: add GxB_FC32 and GxB_FC64
        LAGRAPH_ERROR ("Type not supported", GrB_INVALID_VALUE) ;
    }
    typename [72] = '\0' ;

    //--------------------------------------------------------------------------
    // write the header in ascii
    //--------------------------------------------------------------------------

    // The header is informational only, for "head" command, so the file can
    // be visually inspected.

    char version [LEN] ;
    snprintf (version, LEN, "%d.%d.%d (LAGraph DRAFT)",
        GxB_IMPLEMENTATION_MAJOR,
        GxB_IMPLEMENTATION_MINOR,
        GxB_IMPLEMENTATION_SUB) ;
    version [25] = '\0' ;

    char user [LEN] ;
    for (int k = 0 ; k < LEN ; k++) user [k] = ' ' ;
    user [0] = '\n' ;
    if (comments != NULL)
    {
        strncpy (user, comments, 220) ;
    }
    user [220] = '\0' ;

    char header [LAGRAPH_BIN_HEADER] ;
    int32_t len = snprintf (header, LAGRAPH_BIN_HEADER,
        "SuiteSparse:GraphBLAS matrix\nv%-25s\n"
        "nrows:  %-18" PRIu64 "\n"
        "ncols:  %-18" PRIu64 "\n"
        "nvec:   %-18" PRIu64 "\n"
        "nvals:  %-18" PRIu64 "\n"
        "format: %-8s\n"
        "size:   %-18" PRIu64 "\n"
        "type:   %-72s\n"
        "%-220s\n\n",
        version, nrows, ncols, nvec, nvals, fmt_string, (uint64_t) typesize,
        typename, user) ;

    // printf ("header len %d\n", len) ;
    for (int32_t k = len ; k < LAGRAPH_BIN_HEADER ; k++) header [k] = ' ' ;
    header [LAGRAPH_BIN_HEADER-1] = '\0' ;
    FWRITE (header, sizeof (char), LAGRAPH_BIN_HEADER) ;

    //--------------------------------------------------------------------------
    // write the scalar content
    //--------------------------------------------------------------------------

    FWRITE (&fmt,      sizeof (GxB_Format_Value), 1) ;
    FWRITE (&kind,     sizeof (int32_t), 1) ;
    FWRITE (&hyper,    sizeof (double), 1) ;
    FWRITE (&nrows,    sizeof (GrB_Index), 1) ;
    FWRITE (&ncols,    sizeof (GrB_Index), 1) ;
    FWRITE (&nonempty, sizeof (int64_t), 1) ;
    FWRITE (&nvec,     sizeof (GrB_Index), 1) ;
    FWRITE (&nvals,    sizeof (GrB_Index), 1) ;
    FWRITE (&typecode, sizeof (int32_t), 1) ;
    FWRITE (&typesize, sizeof (size_t), 1) ;

    //--------------------------------------------------------------------------
    // write the array content
    //--------------------------------------------------------------------------

    if (is_hyper)
    {
        FWRITE (Ap, sizeof (GrB_Index), Ap_size) ;
        FWRITE (Ah, sizeof (GrB_Index), Ah_size) ;
        FWRITE (Ai, sizeof (GrB_Index), Ai_size) ;
    }
    else if (is_sparse)
    {
        FWRITE (Ap, sizeof (GrB_Index), Ap_size) ;
        FWRITE (Ai, sizeof (GrB_Index), Ai_size) ;
    }
    else if (is_bitmap)
    {
        FWRITE (Ab, sizeof (int8_t), Ab_size) ;
    }

    FWRITE (Ax, typesize, Ax_size) ;
    fclose (f) ;

    //--------------------------------------------------------------------------
    // re-import the matrix
    //--------------------------------------------------------------------------

    if (fmt == GxB_BY_COL && is_hyper)
    {
        // hypersparse CSC
        #if GxB_IMPLEMENTATION >= GxB_VERSION (4,0,1)
        LAGRAPH_OK (GxB_Matrix_import_HyperCSC (A, type, nrows, ncols,
            &Ap, &Ah, &Ai, &Ax, Ap_size, Ah_size, Ai_size, Ax_size,
            nvec,jumbled, NULL)) ;
        #elif GxB_IMPLEMENTATION == GxB_VERSION (4,0,0)
        LAGRAPH_OK (GxB_Matrix_import_HyperCSC (A, type, nrows, ncols, Ai_size,
            jumbled, nonempty, Ah_size, &Ah, &Ap, &Ai, &Ax, NULL)) ;
        #else
        LAGRAPH_OK (GxB_Matrix_import_HyperCSC (A, type, nrows, ncols, Ai_size,
            nonempty, Ah_size, &Ah, &Ap, &Ai, &Ax, NULL)) ;
        #endif
    }
    else if (fmt == GxB_BY_ROW && is_hyper)
    {
        // hypersparse CSR
        #if GxB_IMPLEMENTATION >= GxB_VERSION (4,0,1)
        LAGRAPH_OK (GxB_Matrix_import_HyperCSR (A, type, nrows, ncols,
            &Ap, &Ah, &Ai, &Ax, Ap_size, Ah_size, Ai_size, Ax_size,
            nvec,jumbled, NULL)) ;
        #elif GxB_IMPLEMENTATION == GxB_VERSION (4,0,0)
        LAGRAPH_OK (GxB_Matrix_import_HyperCSR (A, type, nrows, ncols, Ai_size,
            jumbled, nonempty, Ah_size, &Ah, &Ap, &Ai, &Ax, NULL)) ;
        #else
        LAGRAPH_OK (GxB_Matrix_import_HyperCSR (A, type, nrows, ncols, Ai_size,
            nonempty, Ah_size, &Ah, &Ap, &Ai, &Ax, NULL)) ;
        #endif
    }
    else if (fmt == GxB_BY_COL && is_sparse)
    {
        // standard CSC
        #if GxB_IMPLEMENTATION >= GxB_VERSION (4,0,1)
        LAGRAPH_OK (GxB_Matrix_import_CSC (A, type, nrows, ncols,
            &Ap, &Ai, &Ax, Ap_size, Ai_size, Ax_size, jumbled, NULL)) ;
        #elif GxB_IMPLEMENTATION == GxB_VERSION (4,0,0)
        LAGRAPH_OK (GxB_Matrix_import_CSC (A, type, nrows, ncols, Ai_size,
            jumbled, nonempty, &Ap, &Ai, &Ax, NULL)) ;
        #else
        LAGRAPH_OK (GxB_Matrix_import_CSC (A, type, nrows, ncols, Ai_size,
            nonempty, &Ap, &Ai, &Ax, NULL)) ;
        #endif
    }
    else if (fmt == GxB_BY_ROW && is_sparse)
    {
        // standard CSR
        #if GxB_IMPLEMENTATION >= GxB_VERSION (4,0,1)
        LAGRAPH_OK (GxB_Matrix_import_CSR (A, type, nrows, ncols,
            &Ap, &Ai, &Ax, Ap_size, Ai_size, Ax_size, jumbled, NULL)) ;
        #elif GxB_IMPLEMENTATION == GxB_VERSION (4,0,0)
        LAGRAPH_OK (GxB_Matrix_import_CSR (A, type, nrows, ncols, Ai_size,
            jumbled, nonempty, &Ap, &Ai, &Ax, NULL)) ;
        #else
        LAGRAPH_OK (GxB_Matrix_import_CSR (A, type, nrows, ncols, Ai_size,
            nonempty, &Ap, &Ai, &Ax, NULL)) ;
        #endif
    }
    else if (fmt == GxB_BY_COL && is_bitmap)
    {
        // bitmap by col
        #if GxB_IMPLEMENTATION >= GxB_VERSION (4,0,1)
        LAGRAPH_OK (GxB_Matrix_import_BitmapC (A, type, nrows, ncols,
            &Ab, &Ax, Ab_size, Ax_size, nvals, NULL)) ;
        #endif
    }
    else if (fmt == GxB_BY_ROW && is_bitmap)
    {
        // bitmap by row
        #if GxB_IMPLEMENTATION >= GxB_VERSION (4,0,1)
        LAGRAPH_OK (GxB_Matrix_import_BitmapR (A, type, nrows, ncols,
            &Ab, &Ax, Ab_size, Ax_size, nvals, NULL)) ;
        #endif
    }
    else if (fmt == GxB_BY_COL && is_full)
    {
        // full by col
        #if GxB_IMPLEMENTATION >= GxB_VERSION (4,0,1)
        LAGRAPH_OK (GxB_Matrix_import_FullC (A, type, nrows, ncols,
            &Ax, Ax_size, NULL)) ;
        #endif
    }
    else if (fmt == GxB_BY_ROW && is_full)
    {
        // full by row
        #if GxB_IMPLEMENTATION >= GxB_VERSION (4,0,1)
        LAGRAPH_OK (GxB_Matrix_import_FullR (A, type, nrows, ncols,
            &Ax, Ax_size, NULL)) ;
        #endif
    }
    else
    {
        LAGRAPH_ERROR ("unknown", GrB_INVALID_VALUE) ;
    }

    #if GxB_IMPLEMENTATION >= GxB_VERSION (4,0,1)
    LAGRAPH_OK (GxB_set (*A, GxB_HYPER_SWITCH, hyper)) ;
    #else
    LAGRAPH_OK (GxB_set (*A, GxB_HYPER, hyper)) ;
    #endif

    return (GrB_SUCCESS) ;
}

