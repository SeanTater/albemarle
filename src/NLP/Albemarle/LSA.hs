{-# LANGUAGE OverloadedStrings, NoImplicitPrelude, BangPatterns #-}
module NLP.Albemarle.LSA
    ( apply
    , discover
    , sparseStochasticTruncatedSVD
    , stochasticTruncatedSVD
    , standardLSA
    ) where
import NLP.Albemarle
import ClassyPrelude hiding ((<>)) -- this <> is for semigroups (usually better)
import Data.Monoid ((<>)) -- ... but HMatrix only has Monoid
import qualified NLP.Albemarle.Sparse as Sparse
import qualified Data.Vector.Unboxed as Vec
import qualified Data.HashMap.Strict as HashMap
import Numeric.LinearAlgebra (Matrix, tr) -- transpose
import qualified Numeric.LinearAlgebra as HMatrix
import qualified Data.Text.Format as Format
import qualified System.IO.Streams as Streams
import qualified Criterion.Main
import System.IO.Streams (Generator, InputStream, OutputStream)

type LSAModel = Matrix Double

ffor = flip map

apply :: ()
apply = ()

-- | Generate an LSA model from an input stream of sparse vectors.
--   In the current implementation it all needs to be in memory. That can b
discover :: ()
discover = ()

sparseLSA :: Int -> InputStream (Document) -> IO LSAModel
sparseLSA num_topics docs = do
  doclist <- Streams.toList docs
  (u, _, _) <- sparseStochasticTruncatedSVD num_topics 2 $ Sparse.fromDocuments doclist
  return u

-- | Stochastic SVD. See Halko, Martinsson, and Tropp, 2010 for an explanation
sparseStochasticTruncatedSVD :: Int -> Int -> Sparse.SparseMatrix -> IO (Matrix Double, HMatrix.Vector Double, Matrix Double)
sparseStochasticTruncatedSVD top_vectors num_iterations original = do
  let (height, width) = Sparse.size original
      k = top_vectors + 10

  -- Stage A
  putStrLn "Stage A\n\tPower Iteration"
  omega <- Sparse.sparse <$> HMatrix.randn width k
  -- In the paper this is (AA^T)\sup{q} A \omega
  let originalT = Sparse.shift $ Sparse.transpose original
  --let y 0 = original `Sparse.mult` omega -- Usually the bottleneck
  --    y n = seq (y (n-1)) $ original `Sparse.mult` originalT `Sparse.mult` y (n-1) -- a little faster
  --bigSparseY <- return $! y num_iterations -- Force evaluation here
  let bigSparseY =
        originalT `Sparse.multCol`
        (originalT `Sparse.multCol`
        (original `Sparse.multCol`
        (originalT `Sparse.multCol`
        (original `Sparse.multCol` omega))))
  putStrLn $ "\tDensifying " ++ tshow (Sparse.size bigSparseY)
  bigY <- return $! Sparse.dense bigSparseY
  putStrLn "\tOrthogonalizing"
  q <- return $! HMatrix.orth bigY -- Typically a slow point (force again)

  -- Stage B
  putStrLn "Stage B"
  let b = Sparse.dense $ ( Sparse.sparse $ tr q ) `Sparse.mult` original
      (uhat, sigma, vt) = HMatrix.thinSVD b
      u = q <> uhat

  return $! (HMatrix.takeColumns top_vectors u,
    HMatrix.subVector 0 top_vectors sigma,
    HMatrix.takeRows top_vectors $ tr vt)


standardLSA :: Int -> InputStream (Document) -> IO LSAModel
standardLSA num_topics sparse_vecs = do
  -- full_mat: [[(word_id, count)]]
  full_mat <-  Streams.map Vec.toList sparse_vecs >>= Streams.toList
  let assoc_mat :: [((Int, Int), Double)]
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
  putStrLn "Stage A\n\tPower Iteration"
  omega <- HMatrix.randn width k
  -- In the paper this is (AA^T)\sup{q} A \omega
  let y 0 = original <> omega -- Usually the bottleneck
      y n = seq (y (n-1)) $ original <> tr original <> y (n-1) -- a little faster
  bigY <- return $! y num_iterations -- Force evaluation here
  putStrLn "\tOrthogonalizing"
  q <- return $! HMatrix.orth bigY -- Typically a slow point (force again)

  -- Stage B
  putStrLn "Stage B"
  let b = tr q <> original
      (uhat, sigma, vt) = HMatrix.thinSVD b
      u = q <> uhat

  return $! (HMatrix.takeColumns top_vectors u,
    HMatrix.subVector 0 top_vectors sigma,
    HMatrix.takeRows top_vectors $ tr vt)
