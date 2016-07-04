{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE TemplateHaskell   #-}
{-# LANGUAGE UnicodeSyntax     #-}
{-# LANGUAGE BangPatterns      #-}
module NLP.Albemarle.LSA
    ( batchLSA
    , LSAModel(..)
    , lsa
    , rebase
    , termvectors
    , topicweights
    , SVD.sparsify
    , SVD.sparseSvd
    ) where
import NLP.Albemarle
import NLP.Albemarle.Dict (Dict)
import qualified NLP.Albemarle.Dict as Dict
import Data.List (cycle)
import ClassyPrelude hiding (Vector)
import qualified Data.Vector.Generic as Vec
import qualified Data.Vector.Unboxed as UVec
import qualified Data.Vector.Storable as SVec
import Foreign.C.Types
import qualified Data.HashMap.Strict as HashMap
import Numeric.LinearAlgebra (Vector, Matrix, tr, diag, (|||), (===)) -- transpose
import qualified Numeric.LinearAlgebra as HMatrix
import qualified Numeric.LinearAlgebra.Devel as HMatrix
import qualified Numeric.LinearAlgebra.SVD.SVDLIBC as SVD
import qualified Data.Text.Format as Format
import qualified System.IO.Streams as Streams
import qualified Criterion.Main
import System.IO.Streams (Generator, InputStream, OutputStream)
import Control.Lens.TH
import Control.Lens.Operators

data LSAModel = LSAModel {
  _topicweights :: !(HMatrix.Vector Double),
  _termvectors :: !(Matrix Double)
} deriving (Show, Eq)
makeLenses ''LSAModel

-- This instance may be a fib. I'm not sure this is actually associative.
-- Even if it is, I imagine there are practical numerical stability concerns.
instance Monoid LSAModel where
  mempty = LSAModel mempty mempty
  mappend left right
    | left == mempty = right
    | right == mempty = left
    | otherwise = LSAModel s v
    where
      v = HMatrix.takeRows target_len v'
      s = Vec.take target_len s'
      target_len = Vec.length $ left^.topicweights
      (s', v') = HMatrix.rightSV
        $ traceShow ("In mappend, after = ", HMatrix.size c) c
      c = traceShow ("In mappend, left = ", HMatrix.size (left^.termvectors),
        " right = ", HMatrix.size (right^.termvectors)) (combine left === combine right)
instance Semigroup LSAModel where
  (<>) = mappend

combine :: LSAModel -> Matrix Double
combine model = diag (model^.topicweights) <> (model^.termvectors)

sparseToCSR :: SparseMatrix -> HMatrix.CSR
sparseToCSR (SparseMatrix width vecs) =
  let
    docs = Vec.fromList <$> (\(SparseVector len wds) -> wds) <$> vecs
    convI = Vec.map fromIntegral . Vec.convert
    counts = Vec.map snd <$> docs :: [UVec.Vector Double]
    concatcounts = Vec.concat counts :: UVec.Vector Double
    storcounts = Vec.convert concatcounts :: SVec.Vector Double
  in HMatrix.CSR {
    HMatrix.csrVals = storcounts,
    HMatrix.csrCols = convI
      $ Vec.concat
      $ Vec.map ((+1).fst)
      <$> docs,
    HMatrix.csrRows = convI
      $ Vec.scanl (+) 1
      $ Vec.fromList
      $ Vec.length
      <$> docs,
    HMatrix.csrNCols = width,
    HMatrix.csrNRows = length docs
  }

-- | Generate an LSA Model with N topics, based on a sparse matrix.
--
--   You can make the sparse matrix with a Dict and some documents.
--   See Dict.asSparseMatrix.
lsa :: Int -- ^ Number of vectors/dimensions in the new space
    -> SparseMatrix -- ^ Sparse representation of the term-document matrix
    -> LSAModel -- ^ A dense representation of the term-topic matrix
lsa top_vectors termdoc = traceShow ("Right after SVD", HMatrix.size vt) $ LSAModel s vt
  where (u, s, vt) = SVD.sparseSvd top_vectors $ sparseToCSR termdoc

-- | Rebase an LSA model made with one Dict to one made with another Dict.
--
--   By using this, you can split a corpus into batches, make Dicts and topic
--   models, then merge the Dicts, rebase the Models to the merged Dict, and
--   then merge the models. It allows working in parallel, and limits memory
--   usage.
rebase :: Dict -> Dict -> LSAModel -> LSAModel
rebase d1 d2 model
  | model == mempty = mempty
  | otherwise = let
  (size, support, indices) = traceShowId $ (
    (Dict.size d1, Dict.size d2),
    ("During Rebase", HMatrix.size $ model^.termvectors, Vec.length $ model^.topicweights),
    Dict.select d1 d2 )
  in LSAModel {
    _termvectors = (HMatrix.¿) (model^.termvectors) indices,
    _topicweights = model^.topicweights
  }


-- | SVD with some transposes, for convenience and speed
-- Normally (U, Sigma, V^T) = svd A, but then the rows are topics and the cols
-- are the documents/words (in U and V). But we use C-style (row major) matrices
-- which means for large matrices, getting the vector of one word or one doc
-- will require reading the whole model from memory. (A terrible waste of cache)
-- So instead we use (U^T, sigma, V) where the rows are the vector embeddings
-- of documents and words instead.
--
-- Prefer lsa (rather than batchLSA) - batchLSA is experimental and may be
-- removed
batchLSA :: Int -> HMatrix.CSR -> (Matrix Double, HMatrix.Vector Double, Matrix Double)
batchLSA top_vectors csr = (tr u, s, tr vt)
  where (u, s, vt) = SVD.sparseSvd top_vectors csr
