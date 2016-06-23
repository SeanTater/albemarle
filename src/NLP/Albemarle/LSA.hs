{-# LANGUAGE OverloadedStrings, NoImplicitPrelude, BangPatterns #-}
module NLP.Albemarle.LSA
    ( batchLSA
    , docsToCSR
    , stochasticTruncatedSVD
    , standardLSA
    , SVD.sparsify
    , SVD.sparseSvd
    ) where
import NLP.Albemarle
import Data.List (cycle)
import ClassyPrelude hiding ((<>)) -- this <> is for semigroups (usually better)
import Data.Monoid ((<>)) -- ... but HMatrix only has Monoid
import qualified NLP.Albemarle.Sparse as Sparse
import qualified Data.Vector.Generic as Vec
import qualified Data.Vector.Unboxed as UVec
import qualified Data.Vector.Storable as SVec
import Foreign.C.Types
import qualified Data.HashMap.Strict as HashMap
import Numeric.LinearAlgebra (Matrix, tr) -- transpose
import qualified Numeric.LinearAlgebra as HMatrix
import qualified Numeric.LinearAlgebra.Devel as HMatrix
import qualified Numeric.LinearAlgebra.SVD.SVDLIBC as SVD
import qualified Data.Text.Format as Format
import qualified System.IO.Streams as Streams
import qualified Criterion.Main
import System.IO.Streams (Generator, InputStream, OutputStream)
import Debug.Trace

type LSAModel = Matrix Double


--sparseLSA :: Int -> InputStream BagOfWords -> IO LSAModel
--sparseLSA num_topics docs = do
--  doclist <- Streams.toList docs
--  let (u, _, _) = batchLSA num_topics $ Sparse.fromDocuments doclist
--  return u
docsToCSR :: Int -> InputStream BagOfWords -> IO HMatrix.CSR
docsToCSR width docstream = do
  docs <- Streams.toList docstream
  let
    convI = Vec.map fromIntegral . Vec.convert
    counts = Vec.map (fromIntegral.snd) <$> docs :: [UVec.Vector Double]
    concatcounts = Vec.concat counts :: UVec.Vector Double
    storcounts = Vec.convert concatcounts :: SVec.Vector Double
  return $ HMatrix.CSR {
    HMatrix.csrVals = storcounts,
    HMatrix.csrCols = convI $ Vec.concat $ Vec.map ((+1).fst) <$> docs,
    HMatrix.csrRows = convI $ Vec.scanl (+) 0 $ Vec.fromList $ Vec.length <$> docs,
    HMatrix.csrNCols = width,
    HMatrix.csrNRows = length docs
  }

-- | SVD with some transposes, for convenience and speed
-- Normally (U, Sigma, V^T) = svd A, but then the rows are topics and the cols
-- are the documents/words (in U and V). But we use C-style (row major) matrices
-- which means for large matrices, getting the vector of one word or one doc
-- will require reading the whole model from memory. (A terrible waste of cache)
-- So instead we use (U^T, sigma, V) where the rows are the vector embeddings
-- of documents and words instead.
batchLSA :: Int -> HMatrix.CSR -> (Matrix Double, HMatrix.Vector Double, Matrix Double)
batchLSA top_vectors csr =
  let (u, s, vt) = SVD.sparseSvd top_vectors csr
  in  (tr u, s, tr vt)

standardLSA :: Int -> InputStream BagOfWords -> IO LSAModel
standardLSA num_topics sparse_vecs = do
  -- full_mat: [[(word_id, count)]]
  full_mat <-  Streams.map Vec.toList sparse_vecs >>= Streams.toList
  let ffor = flip map
      assoc_mat :: [((Int, Int), Double)]
      assoc_mat = mconcat $
        ffor (zip [0..] full_mat) $ \(doc_id, word_n_count) ->
            ffor word_n_count $ \(word_id, count) ->
                ((doc_id, word_id), fromIntegral count)
  (u, _, _) <- stochasticTruncatedSVD num_topics 2 $ HMatrix.toDense assoc_mat
  return u

-- Stochastic SVD. See Halko, Martinsson, and Tropp, 2010 for an explanation
stochasticTruncatedSVD :: Int -> Int -> Matrix Double -> IO (Matrix Double, HMatrix.Vector Double, Matrix Double)
stochasticTruncatedSVD top_vectors num_iterations original = do
  let (height, width) = HMatrix.size original
      k = top_vectors + 10

  -- Stage A
  omega <- HMatrix.randn width k
  -- In the paper this is (AA^T)\sup{q} A \omega
  let factors = (take (2*num_iterations) $ cycle [original, tr original]) ++ [original, omega] :: [Matrix Double]
  bigY <- return $! mconcat factors
  q <- return $! HMatrix.orth bigY -- Typically a slow point (force again)

  -- Stage B
  let b = tr q <> original
      (uhat, sigma, vt) = HMatrix.thinSVD b
      u = q <> uhat

  return $! (HMatrix.takeColumns top_vectors u,
    HMatrix.subVector 0 top_vectors sigma,
    HMatrix.takeColumns top_vectors vt)
