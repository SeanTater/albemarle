{-# LANGUAGE OverloadedStrings, NoImplicitPrelude, BangPatterns #-}
module NLP.Albemarle.LSA
    ( batchLSA
    , docsToCSR
    , LSAModel(..)
    , lsa
    , SVD.sparsify
    , SVD.sparseSvd
    ) where
import NLP.Albemarle
import Data.List (cycle)
import ClassyPrelude hiding (Vector)
import qualified NLP.Albemarle.Sparse as Sparse
import qualified Data.Vector.Generic as Vec
import qualified Data.Vector.Unboxed as UVec
import qualified Data.Vector.Storable as SVec
import Foreign.C.Types
import qualified Data.HashMap.Strict as HashMap
import Numeric.LinearAlgebra (Vector, Matrix, tr, diag, (|||)) -- transpose
import qualified Numeric.LinearAlgebra as HMatrix
import qualified Numeric.LinearAlgebra.Devel as HMatrix
import qualified Numeric.LinearAlgebra.SVD.SVDLIBC as SVD
import qualified Data.Text.Format as Format
import qualified System.IO.Streams as Streams
import qualified Criterion.Main
import System.IO.Streams (Generator, InputStream, OutputStream)
import Debug.Trace

data LSAModel = LSAModel !(Matrix Double) !(HMatrix.Vector Double)
  deriving (Show, Eq)

-- This instance may be a fib. I'm not sure this is actually associative.
-- Even if it is, I imagine there are practical numerical stability concerns.
instance Monoid LSAModel where
  mempty = LSAModel mempty mempty
  mappend left right = LSAModel u s
    where (u, s) = HMatrix.leftSV (combine left ||| combine right)
instance Semigroup LSAModel where
  (<>) = mappend

combine :: LSAModel -> Matrix Double
combine (LSAModel m s) = m <> diag s

docsToCSR :: Int -> [SparseVector] -> HMatrix.CSR
docsToCSR width docs_lists =
  let
    docs = Vec.fromList <$> docs_lists
    convI = Vec.map fromIntegral . Vec.convert
    counts = Vec.map (fromIntegral.snd) <$> docs :: [UVec.Vector Double]
    concatcounts = Vec.concat counts :: UVec.Vector Double
    storcounts = Vec.convert concatcounts :: SVec.Vector Double
  in HMatrix.CSR {
    HMatrix.csrVals = storcounts,
    HMatrix.csrCols = convI $ Vec.concat $ Vec.map ((+1).fst) <$> docs,
    HMatrix.csrRows = convI $ Vec.scanl (+) 0 $ Vec.fromList $ Vec.length <$> docs,
    HMatrix.csrNCols = width,
    HMatrix.csrNRows = length docs
  }

lsa :: Int -- ^ Number of vectors/dimensions in the new space
    -> HMatrix.CSR -- ^ Sparse representation of the term-document matrix
    -> LSAModel -- ^ A dense representation of the term-topic matrix
lsa top_vectors termdoc = LSAModel u s
  where (u, s, vt) = SVD.sparseSvd top_vectors termdoc

-- | SVD with some transposes, for convenience and speed
-- Normally (U, Sigma, V^T) = svd A, but then the rows are topics and the cols
-- are the documents/words (in U and V). But we use C-style (row major) matrices
-- which means for large matrices, getting the vector of one word or one doc
-- will require reading the whole model from memory. (A terrible waste of cache)
-- So instead we use (U^T, sigma, V) where the rows are the vector embeddings
-- of documents and words instead.
batchLSA :: Int -> HMatrix.CSR -> (Matrix Double, HMatrix.Vector Double, Matrix Double)
batchLSA top_vectors csr = (tr u, s, tr vt)
  where (u, s, vt) = SVD.sparseSvd top_vectors csr
