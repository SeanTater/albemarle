{-# LANGUAGE OverloadedStrings #-}
{-# LANGUAGE NoImplicitPrelude #-}
{-# LANGUAGE TemplateHaskell   #-}
{-# LANGUAGE UnicodeSyntax     #-}
{-# LANGUAGE BangPatterns      #-}
module NLP.Albemarle.LSA
    (-- * Creating a model
    lsa
    -- * Managing terms
    , rebase
    -- * Managing topics
    , pad
    , trim
    -- * Internal methods (may be subject to change)
    , SVD.sparsify
    , SVD.sparseSvd
    , batchLSA
    -- * Types (may be subject to change)
    , LSAModel(..)
    , termvectors
    , topicweights
    ) where
import NLP.Albemarle
import NLP.Albemarle.Dict (Dict)
import qualified NLP.Albemarle.Dict as Dict
import ClassyPrelude hiding (Vector)
import qualified Data.Vector.Generic as Vec
import qualified Data.Vector.Unboxed as UVec
import qualified Data.Vector.Storable as SVec
import qualified Data.HashMap.Strict as HashMap
import Numeric.LinearAlgebra (Vector, Matrix, tr, diag, (|||), (===))
import qualified Numeric.LinearAlgebra as HMatrix
import qualified Numeric.LinearAlgebra.Devel as HMatrix
import qualified Numeric.LinearAlgebra.SVD.SVDLIBC as SVD
import qualified System.IO.Streams as Streams
import System.IO.Streams (Generator, InputStream, OutputStream)
import Lens.Micro.TH
import Lens.Micro
import Data.Tuple

-- | An LSA model (The singular values and right singular vectors of truncated
--   SVD on a term-document matrix, where documents are rows and terms columns)
data LSAModel = LSAModel {
  _topicweights :: !(HMatrix.Vector Double), -- ^ Topic weights
  _termvectors :: !(Matrix Double) -- ^ Rows are topics, columns are terms
} deriving (Show, Eq)
makeLenses ''LSAModel

-- This instance may be a fib. I'm not sure this is actually associative.
-- Even if it is, I imagine there are practical numerical stability concerns.
-- TODO: In particular, I think we need to weight the left and weight according
-- to how many documents they represent.
instance Monoid LSAModel where
  mempty = LSAModel mempty mempty
  mappend left right
    | left == mempty = right
    | right == mempty = left
    | otherwise = trim target_len $ pad target_len $ LSAModel s v
    where
      target_len = max (topicCount left) (topicCount right)
      (s, v) = HMatrix.rightSV $ combine left === combine right
instance Semigroup LSAModel where
  (<>) = mappend

-- | Truncate an LSAModel to a specific number of topics, if there are too many
trim :: Int -- ^ The maximum number of topics the model should have (inclusive)
     -> LSAModel -> LSAModel
trim count model
  | topicCount model <= count = model
  | otherwise = LSAModel
    (Vec.take count $ model^.topicweights)
    (HMatrix.takeRows count $ model^.termvectors)

-- | Pad an LSAModel with extra empty topics of there are too few
pad :: Int -- ^ The minimum number of topics the model should have (inclusive)
    -> LSAModel -> LSAModel
pad count model
  | topicCount model >= count = model
  | otherwise = let
    (height, width) = HMatrix.size $ model^.termvectors
    missing = count - height
    in LSAModel
      (model^.topicweights <> Vec.replicate missing 0)
      (model^.termvectors === HMatrix.konst 0 (missing, width))

-- | Multiply the vectors and values of an SVD to make one matrix
combine :: LSAModel -> Matrix Double
combine model = diag (model^.topicweights) <> (model^.termvectors)

sparseToCSR :: SparseMatrix -> HMatrix.CSR
sparseToCSR (SparseMatrix width vecs) =
  let
    docs = (Vec.fromList . (\ (SparseVector len wds) -> wds) <$> vecs)
    convI = Vec.map fromIntegral . Vec.convert
    counts = Vec.map snd <$> docs         :: [UVec.Vector Double]
    concatcounts = Vec.concat counts      ::  UVec.Vector Double
    storcounts = Vec.convert concatcounts ::  SVec.Vector Double
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
lsa top_vectors termdoc =
  -- This astounds me, but the number of singular values may not match the
  -- number of singular vectors. We have to balance before padding.
  pad top_vectors $ case compare missing_values 0 of
    LT -> LSAModel (Vec.take top_vectors s) vt
    EQ -> LSAModel s vt
    GT -> LSAModel (s <> Vec.replicate missing_values 0) vt
  where
    (u, s, vt) = SVD.sparseSvd top_vectors $ sparseToCSR termdoc
    missing_values = fst (HMatrix.size vt) - Vec.length s

-- | Rebase an LSA model made with one Dict to one made with another Dict.
--
--   By using this, you can split a corpus into batches, make Dicts and topic
--   models, then merge the Dicts, rebase the Models to the merged Dict, and
--   then merge the models. It allows working in parallel, and limits memory
--   usage.
rebase :: Dict -> Dict -> LSAModel -> LSAModel
rebase d1 d2 model
  | model == mempty = mempty
  | otherwise = LSAModel {
    _termvectors = (HMatrix.¿) (model^.termvectors) $ Dict.select d1 d2,
    _topicweights = model^.topicweights
  }

-- | Get the number of topics in a model
topicCount :: LSAModel -> Int
topicCount model = Vec.length $ model^.topicweights


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
